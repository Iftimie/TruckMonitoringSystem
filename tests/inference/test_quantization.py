import os.path as osp
import cv2
from truckms.api import FrameDatapoint, PredictionDatapoint
from truckms.inference.neural import create_model, compute, plot_detections
from pprint import pprint
from torch.quantization.fuse_modules import _get_module
from truckms.inference.quantization import *
from torch.quantization.fuse_modules import fuse_modules
import pytest


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


def size_of_model(model):
    for name, parameter in model.named_parameters():
        print(name)
    # size = os.path.getsize("temp.p")/1e6
    # os.remove('temp.p')
    return 10


@pytest.mark.skip(reason="I will just keep this test here. Maybe I will come back to it")
def test_quantiaztion():
    with torch.no_grad():
        test_image = osp.join(osp.dirname(__file__), 'data', 'test_image.PNG')
        test_image = cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_BGR2RGB)
        input_images = [FrameDatapoint(test_image, 1)] * 5
        model = create_model()
        model = model.to('cpu')
        unoptimized_model_size = size_of_model(model)
        num_evaluations = 1

        model.backbone.fpn = QuantizedFeaturePyramidNetwork(model.backbone.fpn)
        start = time.time()
        for i in range(num_evaluations):
            expected_predictions = list(compute(input_images, model, cdevice='cpu'))
        end = time.time()
        unoptimized = (end - start) / num_evaluations

        model = create_model(conf_thr=0.1)
        model = model.to('cpu')
        modules_to_fuse = get_modules_to_fuse(model)
        replace_frozenbatchnorm_batchnorm(model)
        model.eval()
        fuse_modules(model, modules_to_fuse, inplace=True, fuser_func=custom_fuse_func)

        def run_fn(model, run_agrs):
            return compute(input_images, model)

        from torch.quantization.QConfig import default_qconfig
        from torch.quantization.default_mappings import DEFAULT_MODULE_MAPPING
        from torch.quantization.quantize import prepare, propagate_qconfig_
        import torch.nn.intrinsic as nni
        import itertools

        for child in model.modules():
            if isinstance(child, nn.ReLU):
                child.inplace = False

        # TODO i removed the linear layers because they were too complicated for quantization. too much logic
        qconfig_spec = dict(zip({nn.Conv2d, nni.ConvReLU2d, nn.ReLU}, itertools.repeat(default_qconfig)))
        propagate_qconfig_(model.backbone, qconfig_spec)
        model.eval()
        model = torch.quantization.quantize(model, run_fn=run_fn, run_args={}, mapping=DEFAULT_MODULE_MAPPING)
        # model = torch.quantization.quantize_dynamic(
        #     model,qconfig_spec=, dtype=torch.qint8,mapping=DEFAULT_MODULE_MAPPING
        # )
        print(model)
        model.transform = QuantizedGeneralizedRCNNTransform(model.transform)
        model.backbone.fpn = QuantizedFeaturePyramidNetwork(model.backbone.fpn)
        # model.rpn = QuantizedRegionProposalNetwork(model.rpn)
        optimized_model_size = size_of_model(model)
        model = model.to('cpu')
        model.eval()
        start = time.time()
        for i in range(num_evaluations):
            actual_predictions = list(compute(input_images, model, cdevice='cpu'))
        end = time.time()
        optimized = (end - start) / num_evaluations
        pprint(actual_predictions[0].pred['boxes'])
        pprint(expected_predictions[0].pred['boxes'])
        # assert optimized < unoptimized
        print("time UNOPTIMIZED VS OPTIMIZED", unoptimized, optimized)
        print("size UNOPTIMIZED VS OPTIMIZED", unoptimized_model_size, optimized_model_size)
        # cv2.imwrite("/home/aiftimie/out.png", next(plot_detections(input_images, actual_predictions)).image)

        # assert optimized_model_size < unoptimized_model_size
        # UNOPTIMIZED VS OPTIMIZED 0.9593331384658813 0.8527213740348816 batch of 5 images
