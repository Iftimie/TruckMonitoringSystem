from truckms.inference.neural import create_model, compute, plot_detections, pred_iter_to_pandas, pandas_to_pred_iter
from truckms.inference.neural import iterable2batch
from truckms.inference.utils import framedatapoint_generator_by_frame_ids2
from truckms.api import FrameDatapoint, PredictionDatapoint
import os.path as osp
import os
import cv2
from truckms.inference.utils import framedatapoint_generator
from itertools import tee


def test_iterable2batch():
    g1 = framedatapoint_generator_by_frame_ids2(video_path=osp.join(osp.dirname(__file__),
                                                                    '..', 'service', 'data', 'cut.mkv'),
                                      frame_ids=[3, 6, 10, 12, 17])
    g2 = iterable2batch(g1, batch_size=2)

    for idx, bfdp in enumerate(g2):
        if idx != 2:
            assert len(bfdp.batch_images) == 2
        else:
            assert len(bfdp.batch_images) == 1


def test_truck_detector():
    test_image = osp.join(osp.dirname(__file__), 'data', 'test_image.PNG')
    test_image = cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_BGR2RGB)
    model = create_model()

    input_images = [FrameDatapoint(test_image, 1)]
    predictions = list(compute(input_images, model))
    assert len(predictions) != 0
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], PredictionDatapoint)
    assert isinstance(predictions[0].pred, dict)
    assert 'labels' in predictions[0].pred
    assert 'scores' in predictions[0].pred
    assert 'boxes' in predictions[0].pred
    # cv2.imshow("image", next(p.plot_detections(input_images, predictions)))
    # cv2.waitKey(0)


# @pytest.mark.skip(reason="depends on local data")
def test_auu_data():
    auu_data_root = r'D:\aau-rainsnow\Hjorringvej\Hjorringvej-2'
    video_files = [osp.join(r, f) for (r, _, fs) in os.walk(auu_data_root) for f in fs if 'avi' in f or 'mkv' in f]
    model = create_model()

    for video_path in video_files:
        image_gen = framedatapoint_generator(video_path, skip=5)
        image_gen1, image_gen2 = tee(image_gen)

        for idx, fdp in enumerate(plot_detections(image_gen1, compute(image_gen2, model))):
            cv2.imshow("image", fdp.image)
            cv2.waitKey(1)
            if idx==5:break

def test_TruckDetector_pred_iter_to_pandas():
    auu_data_root = r'D:\aau-rainsnow\Hjorringvej\Hjorringvej-2'
    video_file = [osp.join(r, f) for (r, _, fs) in os.walk(auu_data_root) for f in fs if 'avi' in f or 'mkv' in f][0]
    #file 'Hjorringvej\\Hjorringvej-2\\cam1.mkv' has 6000 frames
    model = create_model(max_operating_res=320)
    image_gen = framedatapoint_generator(video_file, skip=6000//30)
    image_gen1, image_gen2 = tee(image_gen)

    pred_gen = compute(image_gen1, model)

    df = pred_iter_to_pandas(pred_gen)

    pred_gen_from_df = pandas_to_pred_iter(df)

    for idx, fdp in enumerate(plot_detections(image_gen2, pred_gen_from_df)):
        cv2.imshow("image", fdp.image)
        cv2.waitKey(1)
        if idx==5:break


from torchvision.models.resnet import Bottleneck
from torch.quantization.fuse_modules import fuse_conv_bn_relu, fuse_conv_bn
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

    def forward(self, x):
        identity = x
        out = self.conv_bn_relu1(x)
        out = self.conv_bn_relu2(out)
        out = self.conv_bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

    def fuse(self):
        self.conv_bn_relu1 = fuse_conv_bn_relu(self.conv1, self.bn1, self.relu)
        self.conv_bn_relu2 = fuse_conv_bn_relu(self.conv2, self.bn2, self.relu)
        self.conv_bn3 = fuse_conv_bn(self.conv3, self.bn3)


def create_fusedbottleneck_from_bottle_neck(btn: Bottleneck) -> FusedBottleneck:
    return FusedBottleneck(btn)


def replace_frozenbatchnorm_batchnorm(model):
    for k in model._modules:
        if isinstance(model._modules[k], FrozenBatchNorm2d):
            fbn = model._modules[k]
            model._modules[k] = nn.BatchNorm2d(fbn.weight.shape[0])
            model._modules[k].load_state_dict(fbn.state_dict())
        else:
            replace_frozenbatchnorm_batchnorm(model._modules[k])

def test_replace_frozenbatchnorm_batchnorm():
    test_image = osp.join(osp.dirname(__file__), 'data', 'test_image.PNG')
    test_image = cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_BGR2RGB)
    input_images = [FrameDatapoint(test_image, 1)]
    model = create_model()
    expected_predictions = list(compute(input_images, model))

    replace_frozenbatchnorm_batchnorm(model)
    for child in model.modules():
        assert not isinstance(child, FrozenBatchNorm2d)
    model = model.eval().to('cuda')
    
    actual_predictions = list(compute(input_images, model))
    assert len(actual_predictions) == len(expected_predictions)
    assert (actual_predictions[0].pred['boxes'] == expected_predictions[0].pred['boxes']).all()


import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
def get_modules_to_fuse(module, parentmodule = ""):
    modules_to_fuse = []

    modules = module._modules
    module_keys = list(modules.keys())

    separator = "" if parentmodule == "" else "."

    i = 0
    while i < len(module_keys):
        key = module_keys[i]
        if isinstance(modules[key], Bottleneck):
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

from pprint import pprint
from torch.quantization.fuse_modules import _get_module
def test_get_models_to_fuse():
    model = create_model(max_operating_res=320)
    model = model.to('cpu')
    model.eval()
    replace_frozenbatchnorm_batchnorm(model)
    expected_layer_names = [['backbone.body.conv1', 'backbone.body.bn1', 'backbone.body.relu'],
 ['backbone.body.layer1.0'],
 ['backbone.body.layer1.1'],
 ['backbone.body.layer1.2'],
 ['backbone.body.layer2.0'],
 ['backbone.body.layer2.1'],
 ['backbone.body.layer2.2'],
 ['backbone.body.layer2.3'],
 ['backbone.body.layer3.0'],
 ['backbone.body.layer3.1'],
 ['backbone.body.layer3.2'],
 ['backbone.body.layer3.3'],
 ['backbone.body.layer3.4'],
 ['backbone.body.layer3.5'],
 ['backbone.body.layer4.0'],
 ['backbone.body.layer4.1'],
 ['backbone.body.layer4.2']]
    modules_to_fuse = get_modules_to_fuse(model)
    for i, j in zip(expected_layer_names, modules_to_fuse):
        for x, z in zip(i, j):
            assert x == z
    assert isinstance(_get_module(model, modules_to_fuse[0][0]), nn.Conv2d)
    assert isinstance(_get_module(model, modules_to_fuse[0][1]), nn.BatchNorm2d)
    assert isinstance(_get_module(model, modules_to_fuse[0][2]), nn.ReLU)
    for module_list in modules_to_fuse[1:]:
        assert isinstance(_get_module(model, module_list[0]), Bottleneck)


from torch.quantization.fuse_modules import fuse_known_modules
import torch
def custom_fuse_func(mod_list):
    r"""
    Similar to torch.quantization.fuse_modules.fuse_known_modules, but also has the Bottleneck module.
    It only adds a new feature of fusing bootleneck module. If there are other types, it calls fuse_known_modules
    """

    OP_LIST_TO_FUSER_METHOD = {
        (Bottleneck,): create_fusedbottleneck_from_bottle_neck,
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


from torch.quantization.fuse_modules import fuse_modules
def test_fusing():
    test_image = osp.join(osp.dirname(__file__), 'data', 'test_image.PNG')
    test_image = cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_BGR2RGB)
    input_images = [FrameDatapoint(test_image, 1)]
    model = create_model()
    expected_predictions = list(compute(input_images, model))

    model = model.to('cpu')
    modules_to_fuse = get_modules_to_fuse(model)
    replace_frozenbatchnorm_batchnorm(model)
    model.eval()
    fuse_modules(model, modules_to_fuse, inplace=True, fuser_func=custom_fuse_func)
    model = model.to('cuda')
    actual_predictions = list(compute(input_images, model))
    assert len(expected_predictions) == len(actual_predictions)
    assert (expected_predictions[0].pred['boxes'] == actual_predictions[0].pred['boxes']).all()
    assert abs((expected_predictions[0].pred['scores'] - actual_predictions[0].pred['scores'])).sum() < 0.1


import torch.quantization
import time
def test_quantiaztion():
    test_image = osp.join(osp.dirname(__file__), 'data', 'test_image.PNG')
    test_image = cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_BGR2RGB)
    input_images = [FrameDatapoint(test_image, 1)] * 5
    model = create_model()
    num_evaluations = 100

    start = time.time()
    for i in range(num_evaluations):
        expected_predictions = list(compute(input_images, model))
    end = time.time()
    unoptimized = (end-start)/num_evaluations

    model = model.to('cpu')
    modules_to_fuse = get_modules_to_fuse(model)
    replace_frozenbatchnorm_batchnorm(model)
    model.eval()
    fuse_modules(model, modules_to_fuse, inplace=True, fuser_func=custom_fuse_func)

    def run_fn(model, run_agrs):
        return compute(input_images, model)

    model = torch.quantization.quantize(model, run_fn=run_fn, run_args={})
    model = model.to('cuda')
    model.eval()
    start = time.time()
    for i in range(num_evaluations):
        actual_predictions = list(compute(input_images, model))
    end = time.time()
    optimized = (end-start)/num_evaluations
    pprint (actual_predictions[0].pred['boxes'])
    pprint (expected_predictions[0].pred['boxes'])
    assert optimized < unoptimized
    print ("UNOPTIMIZED VS OPTIMIZED", unoptimized, optimized)
    # UNOPTIMIZED VS OPTIMIZED 0.9593331384658813 0.8527213740348816 batch of 5 images
