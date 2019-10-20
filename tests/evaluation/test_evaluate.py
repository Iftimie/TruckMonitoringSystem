from truckms.evaluation.evaluate import download_data_if_not_exists, get_dataset, gen_cocoitem2datapoints, \
    target_iter_to_pandas, gen_cocoitem2targetdp, gen_cocoitem2framedp, target_pred_iter_to_pandas
from truckms.api import TargetDatapoint, FrameDatapoint, coco_val_2017_names, model_class_names
from truckms.inference.neural import plot_detections
from truckms.evaluation.evaluate import compute_iou_det_ann_df, compute_stats
import os
import os.path as osp
import platform
import json
from itertools import tee
from typing import Iterable, Tuple
import pandas as pd
import numpy as np
import mock
from truckms.inference.visuals import plot_over_image
from torchvision.datasets import CocoDetection
from truckms.evaluation.evaluate import plot_targets
import cv2


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


@mock.patch.object(CocoDetection, "__len__")
def test_gen_cocoitem2datapoints(mock_some_obj_some_method):
    mock_some_obj_some_method.return_value = 100

    if platform.system() == "Linux":
        datalake_path = r"/data1/workspaces/aiftimie/tms/tms_data"
    else:
        datalake_path = r"D:\tms_data"
    download_data_if_not_exists(datalake_path)
    coco_dset = get_dataset(datalake_path)
    g = gen_cocoitem2datapoints(coco_dset)
    for fdp, tdp in g:
        dummy_pred = tdp.target
        dummy_pred['scores'] = np.zeros_like(dummy_pred['labels'])
        dummy_pred['obj_id'] = np.zeros_like(dummy_pred['labels'])
        image = plot_over_image(fdp.image, dummy_pred)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        assert isinstance(fdp, FrameDatapoint)
        assert isinstance(tdp, TargetDatapoint)


@mock.patch.object(CocoDetection, "__len__")
def test_target_iter_to_pandas(mock_some_obj_some_method):
    mock_some_obj_some_method.return_value = 100

    if platform.system() == "Linux":
        datalake_path = r"/data1/workspaces/aiftimie/tms/tms_data"
    else:
        datalake_path = r"D:\tms_data"
    download_data_if_not_exists(datalake_path)

    coco_dset = get_dataset(datalake_path)

    g = gen_cocoitem2datapoints(coco_dset)
    g2 = gen_cocoitem2targetdp(g)
    df = target_iter_to_pandas(g2)


from truckms.inference.neural import create_model_efficient, compute, create_model
from functools import partial
@mock.patch.object(CocoDetection, "__len__")
def test_target_pred_iter_to_pandas(mock_some_obj_some_method):
    mock_some_obj_some_method.return_value = 100

    if platform.system() == "Linux":
        datalake_path = r"/data1/workspaces/aiftimie/tms/tms_data"
    else:
        datalake_path = r"D:\tms_data"
    download_data_if_not_exists(datalake_path)
    coco_dset = get_dataset(datalake_path)

    g_tdp_fdp_1: Iterable[Tuple[FrameDatapoint, TargetDatapoint]]
    g_tdp_fdp_2: Iterable[Tuple[FrameDatapoint, TargetDatapoint]]
    g_tdp_fdp_1, g_tdp_fdp_2 = tee(gen_cocoitem2datapoints(coco_dset))
    g_tdp = gen_cocoitem2targetdp(g_tdp_fdp_1)
    g_fdp = gen_cocoitem2framedp(g_tdp_fdp_2)


    model = create_model_efficient(model_creation_func=partial(create_model, max_operating_res=800))
    g_pred = compute(g_fdp, model, batch_size=5, filter_classes=model_class_names)

    df_pred, df_target = target_pred_iter_to_pandas(g_tdp, g_pred)
    if platform.system() == "Linux":
        df_pred.to_csv("/data1/workspaces/aiftimie/tms/tms_experiments/pandas_dataframes/coco_pred.csv")
        df_target.to_csv("/data1/workspaces/aiftimie/tms/tms_experiments/pandas_dataframes/coco_target.csv")
    else:
        pass


def test_dataset_by_frame_ids():
    frame_ids= set([724, 1532, 5037, 5992, 6040, 6723, 7088, 7386, 7977, 8762, 9769, 9891])
    if platform.system() == "Linux":
        datalake_path = r"/data1/workspaces/aiftimie/tms/tms_data"
    else:
        datalake_path = r"D:\tms_data"
    download_data_if_not_exists(datalake_path)
    coco_dset = get_dataset(datalake_path)

    g_tdp_fdp_1: Iterable[Tuple[FrameDatapoint, TargetDatapoint]]
    g_tdp_fdp_2: Iterable[Tuple[FrameDatapoint, TargetDatapoint]]
    g_tdp_fdp_1, g_tdp_fdp_2 = tee(gen_cocoitem2datapoints(coco_dset, frame_ids))
    g_tdp = gen_cocoitem2targetdp(g_tdp_fdp_1)
    g_fdp_1, g_fdp_2, g_fdp_3 = tee(gen_cocoitem2framedp(g_tdp_fdp_2),3)

    model = create_model_efficient(model_creation_func=partial(create_model, max_operating_res=800))
    g_pred = compute(g_fdp_1, model, batch_size=5, filter_classes=model_class_names)
    for fdp_pred, fdp_gt in zip(plot_detections(g_fdp_2, g_pred), plot_targets(g_fdp_3, g_tdp)):
        cv2.imshow("image_pred", fdp_pred.image)
        cv2.imshow("image_gt", fdp_gt.image)
        cv2.waitKey(0)


def test_compute():
    df_pred = pd.read_csv(osp.join(osp.dirname(__file__), 'data', 'coco_pred.csv'))
    df_target = pd.read_csv(osp.join(osp.dirname(__file__), 'data', 'coco_target.csv'))

    # TODO currently we have databases with images that have at least one annotation. in the future I have to make sure that in the dataframe are nan rows for each frame, even tough there are no annotations

    label_dataframes = compute_iou_det_ann_df(df_target, df_pred, ann_lbl_col='target.label', det_lbl_col='label')

    # for k in label_dataframes:
    for k in ['car']:
        joined_df = label_dataframes[k][0]
        ann_df = label_dataframes[k][1]
        det_df = label_dataframes[k][2]
        compute_stats(joined_df, ann_df, det_df)
