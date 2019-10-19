from truckms.evaluation.evaluate import download_data_if_not_exists, get_dataset, gen_cocoitem2datapoints, \
    target_iter_to_pandas, gen_cocoitem2targetdp
from truckms.api import TargetDatapoint, FrameDatapoint, coco_val_2017_names
import os
import platform
import json


def test_download_data_if_not_exists(tmpdir):
    download_data_if_not_exists(tmpdir)
    assert os.path.exists(os.path.join(tmpdir, "coco_val_2017"))
    assert os.path.exists(os.path.join(tmpdir, "coco_val_2017", "val2017.zip"))
    assert os.path.exists(os.path.join(tmpdir, "coco_val_2017", "annotations_trainval2017.zip"))
    assert os.path.exists(os.path.join(tmpdir, "coco_val_2017", "val2017"))
    assert os.path.exists(os.path.join(tmpdir, "coco_val_2017", "annotations"))
    assert len(os.listdir(os.path.join(tmpdir, "coco_val_2017", "val2017"))) == 5000
    assert len(os.listdir(os.path.join(tmpdir, "coco_val_2017", "annotations"))) == 6
    json_file = os.path.join(tmpdir, "coco_val_2017", "annotations", "instances_val2017.json")
    if json_file is not None:
        with open(json_file, 'r') as COCO:
            js = json.loads(COCO.read())
            assert { item['id'] :item['name'] for item in js['categories']} == coco_val_2017_names


def test_get_dataset():
    if platform.system() == "Linux":
        datalake_path = r"D:\tms_data"
    else:
        datalake_path = r"/data1/workspaces/aiftimie/tms/tms_data"
    download_data_if_not_exists(datalake_path)
    # if error about some mask: https://stackoverflow.com/questions/49311195/how-to-install-coco-pythonapi-in-python3
    coco_dset = get_dataset(datalake_path)
    assert len(coco_dset) == 5000


def test_gen_cocoitem2datapoints():
    if platform.system() == "Linux":
        datalake_path = r"D:\tms_data"
    else:
        datalake_path = r"/data1/workspaces/aiftimie/tms/tms_data"
    download_data_if_not_exists(datalake_path)
    coco_dset = get_dataset(datalake_path)
    g = gen_cocoitem2datapoints(coco_dset)
    for fdp, tdp in g:
        assert isinstance(fdp, FrameDatapoint)
        assert isinstance(tdp, TargetDatapoint)
        break


import mock
from torchvision.datasets import CocoDetection
@mock.patch.object(CocoDetection, "__len__")
def test_target_iter_to_pandas(mock_some_obj_some_method):
    mock_some_obj_some_method.return_value = 100

    if platform.system() == "Linux":
        datalake_path = r"/data1/workspaces/aiftimie/tms/tms_data"
    else:
        datalake_path = r"D:\tms_data"
    download_data_if_not_exists(datalake_path)

    coco_dset = get_dataset(datalake_path)

    g2 = gen_cocoitem2targetdp(coco_dset)
    df = target_iter_to_pandas(g2)
    pass