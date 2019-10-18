from truckms.evaluation.evaluate import download_data_if_not_exists, get_dataset, gen_cocoitem2datapoints
from truckms.api import TargetDatapoint, FrameDatapoint
import os


def test_download_data_if_not_exists(tmpdir):
    download_data_if_not_exists(tmpdir)
    assert os.path.exists(os.path.join(tmpdir, "coco_val_2017"))
    assert os.path.exists(os.path.join(tmpdir, "coco_val_2017", "val2017.zip"))
    assert os.path.exists(os.path.join(tmpdir, "coco_val_2017", "annotations_trainval2017.zip"))
    assert os.path.exists(os.path.join(tmpdir, "coco_val_2017", "val2017"))
    assert os.path.exists(os.path.join(tmpdir, "coco_val_2017", "annotations"))
    assert len(os.listdir(os.path.join(tmpdir, "coco_val_2017", "val2017"))) == 5000
    assert len(os.listdir(os.path.join(tmpdir, "coco_val_2017", "annotations"))) == 6


def test_get_dataset():
    datalake_path = r"D:\tms_data"
    download_data_if_not_exists(datalake_path)
    # if error about some mask: https://stackoverflow.com/questions/49311195/how-to-install-coco-pythonapi-in-python3
    coco_dset = get_dataset(datalake_path)
    assert len(coco_dset) == 5000

def test_gen_cocoitem2datapoints():
    datalake_path = r"D:\tms_data"
    download_data_if_not_exists(datalake_path)
    coco_dset = get_dataset(datalake_path)
    g = gen_cocoitem2datapoints(coco_dset)
    for fdp, tdp in g:
        assert isinstance(fdp, FrameDatapoint)
        assert isinstance(tdp, TargetDatapoint)
        break