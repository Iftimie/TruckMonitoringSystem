from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision.ops.misc import FrozenBatchNorm2d
from torch.quantization.fuse_modules import fuse_conv_bn_relu, fuse_conv_bn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torch import quantize_per_tensor
import torch.quantization
import time

class FusedBottleneck(Bottleneck):
    # this will have to be returned
    def __init__(self, bottleneck_layer):
        assert isinstance(bottleneck_layer, Bottleneck)
        in_planes = bottleneck_layer.conv1.in_channels
        width = bottleneck_layer.conv1.out_channels
        groups = bottleneck_layer.conv2.groups
        stride = bottleneck_layer.conv2.stride
        dilation = bottleneck_layer.conv2.dilation
        planes = bottleneck_layer.conv3.out_channels // Bottleneck.expansion
        base_width = ((width // groups) // planes) * 64
        downsample = bottleneck_layer.downsample
        norm_layer = type(bottleneck_layer.bn1)
        super(FusedBottleneck, self).__init__(in_planes, planes, stride, downsample, groups,base_width, dilation, norm_layer)
        self._modules.update(bottleneck_layer._modules)
        self.fuse()
        del self.conv1
        del self.bn1
        del self.conv2
        del self.bn2
        del self.conv3
        del self.bn3

    def forward(self, x):
        identity = x
        out = self.conv_bn_relu1(x)
        out = self.conv_bn_relu2(out)
        out = self.conv_bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        if identity.is_quantized or out.is_quantized:
            assert identity.q_scale() == out.q_scale() and identity.q_zero_point() == out.q_zero_point() and identity.dtype == out.dtype
            out = torch.quantize_per_tensor(out.dequantize() + identity.dequantize(), identity.q_scale(), identity.q_zero_point(), identity.dtype)
        else:
            out += identity
        out = self.relu(out)
        return out

    def fuse(self):
        self.conv_bn_relu1 = fuse_conv_bn_relu(self.conv1, self.bn1, self.relu)
        self.conv_bn_relu2 = fuse_conv_bn_relu(self.conv2, self.bn2, self.relu)
        self.conv_bn3 = fuse_conv_bn(self.conv3, self.bn3)
        if self.downsample is not None:
            self.downsample = fuse_conv_bn(self.downsample[0], self.downsample[1])


class FusedBasicBlock(BasicBlock):
    # this will have to be returned
    def __init__(self, basicblock_layer):
        assert isinstance(basicblock_layer, BasicBlock)
        in_planes = basicblock_layer.conv1.in_channels
        planes = basicblock_layer.conv1.out_channels
        stride = basicblock_layer.conv1.stride
        downsample = basicblock_layer.downsample
        norm_layer = type(basicblock_layer.bn1)
        super(FusedBasicBlock, self).__init__(inplanes=in_planes, planes=planes, stride=stride, downsample=downsample,  norm_layer=norm_layer)
        self._modules.update(basicblock_layer._modules)
        self.fuse()
        del self.conv1
        del self.bn1
        del self.conv2
        del self.bn2

    def forward(self, x):
        identity = x
        out = self.conv_bn_relu1(x)
        out = self.conv_bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        if identity.is_quantized or out.is_quantized:
            # assert identity.q_scale() == out.q_scale() and identity.q_zero_point() == out.q_zero_point() and identity.dtype == out.dtype
            out = torch.quantize_per_tensor(out.dequantize() + identity.dequantize(), 1/2**8, 0, identity.dtype)
        else:
            out += identity
        out = self.relu(out)
        return out

    def fuse(self):
        self.conv_bn_relu1 = fuse_conv_bn_relu(self.conv1, self.bn1, self.relu)
        self.conv_bn3 = fuse_conv_bn(self.conv2, self.bn2)
        if self.downsample is not None:
            self.downsample = fuse_conv_bn(self.downsample[0], self.downsample[1])


def create_fusedbottleneck_from_bottle_neck(btn: Bottleneck) -> FusedBottleneck:
    return FusedBottleneck(btn)


def create_fusedbasicblock_from_bottle_neck(btn: Bottleneck) -> FusedBottleneck:
    return FusedBasicBlock(btn)


def replace_frozenbatchnorm_batchnorm(model):
    for k in model._modules:
        if isinstance(model._modules[k], FrozenBatchNorm2d):
            fbn = model._modules[k]
            model._modules[k] = nn.BatchNorm2d(fbn.weight.shape[0])
            model._modules[k].load_state_dict(fbn.state_dict())
        else:
            replace_frozenbatchnorm_batchnorm(model._modules[k])

import torch.nn as nn
def get_modules_to_fuse(module, parentmodule = ""):
    modules_to_fuse = []

    modules = module._modules
    module_keys = list(modules.keys())

    separator = "" if parentmodule == "" else "."

    i = 0
    while i < len(module_keys):
        key = module_keys[i]
        if isinstance(modules[key], Bottleneck) or isinstance(modules[key], BasicBlock) :
            modules_to_fuse.append([parentmodule + separator + key])
            i+=1
            continue
        elif isinstance(modules[key], nn.Conv2d) and i + 1 < len(module_keys):
            next_key = module_keys[i + 1]
            if any(isinstance(modules[next_key], cls) for cls in [nn.BatchNorm1d, nn.BatchNorm2d, FrozenBatchNorm2d]):
                tuple_module_keys_to_fuse = [parentmodule + separator + key, parentmodule + separator + next_key]
                if i + 2 < len(module_keys) and isinstance(modules[module_keys[i+2]], nn.ReLU):
                    tuple_module_keys_to_fuse.append(parentmodule + separator + module_keys[i+2])
                    modules_to_fuse.append(tuple_module_keys_to_fuse)
                    i += 3
                    continue
                else:
                    modules_to_fuse.append(tuple_module_keys_to_fuse)
                    i += 2
                    continue
            else:
                i += 1
                continue
        else:
            modules_to_fuse.extend(get_modules_to_fuse(modules[key], parentmodule=parentmodule + separator + key ))
            i += 1
    return modules_to_fuse



from torch.quantization.fuse_modules import fuse_known_modules
import torch
def custom_fuse_func(mod_list):
    r"""
    Similar to torch.quantization.fuse_modules.fuse_known_modules, but also has the Bottleneck module.
    It only adds a new feature of fusing bootleneck module. If there are other types, it calls fuse_known_modules
    """

    OP_LIST_TO_FUSER_METHOD = {
        (Bottleneck,): create_fusedbottleneck_from_bottle_neck,
        (BasicBlock,): create_fusedbasicblock_from_bottle_neck,
    }

    types = tuple(type(m) for m in mod_list)
    fuser_method = OP_LIST_TO_FUSER_METHOD.get(types, None)
    if fuser_method is None:
        return fuse_known_modules(mod_list)
    else:
        new_mod = [None] * len(mod_list)
        new_mod[0] = fuser_method(*mod_list)

        for i in range(1, len(mod_list)):
            new_mod[i] = torch.nn.Identity()
            new_mod[i].training = mod_list[0].training

    return new_mod


class QuantizedGeneralizedRCNNTransform(GeneralizedRCNNTransform):

    def __init__(self, grcnnt: GeneralizedRCNNTransform):
        super(QuantizedGeneralizedRCNNTransform, self).__init__(min_size=grcnnt.min_size, max_size=grcnnt.max_size, image_mean=grcnnt.image_mean, image_std=grcnnt.image_std)

    def forward(self, images, targets=None):
        image_list, targets = super(QuantizedGeneralizedRCNNTransform, self).forward(images, targets)
        image_list.tensors = quantize_per_tensor(image_list.tensors, 1/(2**8), 0, torch.quint8)
        # image_list.image_sizes = image_sizes
        return image_list, targets



from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.models.detection.rpn import RegionProposalNetwork, concat_box_prediction_layers

class QuantizedRegionProposalNetwork(nn.Module):
    def __init__(self, rpn):
        super(QuantizedRegionProposalNetwork, self).__init__()
        self.rpn = rpn

    def forward(self, images, features, targets=None):
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.rpn.head(features)
        anchors = self.rpn.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level = [o[0].numel() for o in objectness]
        objectness = [i.dequantize() for i in objectness]
        pred_bbox_deltas = [i.dequantize() for i in pred_bbox_deltas]
        objectness, pred_bbox_deltas = \
            concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.rpn.training:
            labels, matched_gt_boxes = self.rpn.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.rpn.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.rpn.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses

class QuantizedFeaturePyramidNetwork(nn.Module):
    def __init__(self, fpn):
        super(QuantizedFeaturePyramidNetwork, self).__init__()
        self.fpn = fpn

    def forward(self, x):
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())
        last_inner = self.fpn.inner_blocks[-1](x[-1])
        results = []
        results.append(self.fpn.layer_blocks[-1](last_inner))
        for feature, inner_block, layer_block in zip(
                x[:-1][::-1], self.fpn.inner_blocks[:-1][::-1], self.fpn.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_lateral = inner_block(feature)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")

            if inner_lateral.is_quantized or inner_top_down.is_quantized:
                assert inner_lateral.q_scale() == inner_top_down.q_scale() and inner_lateral.q_zero_point() == inner_top_down.q_zero_point() and inner_lateral.dtype == inner_top_down.dtype
                last_inner = torch.quantize_per_tensor(inner_lateral.dequantize() + inner_top_down.dequantize(), inner_lateral.q_scale(),
                                                inner_lateral.q_zero_point(), inner_lateral.dtype)
                # last_inner = inner_lateral.dequantize() + inner_top_down.dequantize()
            else:
                last_inner = inner_lateral + inner_top_down
            results.insert(0, layer_block(last_inner))

        if self.fpn.extra_blocks is not None:
            results, names = self.fpn.extra_blocks(results, x, names)

        # make it back an OrderedDict
        if last_inner.is_quantized:
            out = OrderedDict([(k, v.dequantize()) for k, v in zip(names, results)])
        else:
            out = OrderedDict([(k, v) for k, v in zip(names, results)])

        # TODO watch here motherfucker that this part is not quantized. the output is not quantized

        return out
