from truckms.evaluation.evaluate import download_data_if_not_exists, get_dataset, gen_cocoitem2datapoints, \
    target_iter_to_pandas, gen_cocoitem2targetdp, gen_cocoitem2framedp, target_pred_iter_to_pandas
from truckms.api import TargetDatapoint, FrameDatapoint, coco_val_2017_names, model_class_names
from truckms.inference.neural import plot_detections
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


def bb_intersection_over_union(boxA, boxB):
    """
    Will compute the IoU between two bounding boxes. Boxes must have x1,y1, x2, y2 (not x,y width height)

    Args:
        boxA: list of 2d coordinates in the image as tuples
        boxB: list of 2d coordinates in the image as tuples

    Return:
        IoU value
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def compute_iou(row, box1_keys=["x1", "y1", "x2", "y2"], box2_keys=["target.x1", "target.y1", "target.x2", "target.y2"]):
    """
    Receives a pandas row (it could also be a dictionary) having the following keys as default (could be other):
    For the detected bounding box:
        ["x1", "y1", "x2", "y2"]
    For the marked bounding box:
        ["target.x1", "target.y1", "target.x2", "target.y2"]

    Args:
        row: pandas row or dictionary having keys for detected bounding box and marked bounding box. some values may be
        None. in that case IoU will be 0

    Return:
        float representing intersection over union
    """
    if not row[box1_keys[0]]:
        return 0
    if not row[box2_keys[0]]:
        return 0
    box1 = [row[k] for k in box1_keys]
    box2 = [row[k] for k in box2_keys]
    return bb_intersection_over_union(box1, box2)


def compute_iou_df(joined_df, iou_tr=0.3):
    """
    Computes IoU for a dataframe that contains both detectins and annotations columns
    """
    joined_df["iou"] = joined_df.apply(compute_iou, axis=1)
    joined_df["iou>iou_tr"] = joined_df["iou"] > iou_tr
    return joined_df


def clean_and_set_index_df(df, index_key='img_id', drop_nan_row_by_columns=None):
    """
    Drops unnecessary columns and sets index for for the dataframe.

    Args:
        df: pandas dataframe (annotations or detections)

    Returns:
        pandas dataframe cleaned and with the index set
    """
    if "Unnamed: 0" in df.columns:
        df = df.drop(["Unnamed: 0"], axis=1)
    df = df.set_index(index_key)
    if drop_nan_row_by_columns is not None:
        df = df.dropna(subset=drop_nan_row_by_columns)
    return df


def join_dataframes(ann_df, det_df):
    """
    Joins annotations dataframe and detections dataframe. The resulting dataframe will have rows for each marking of
    the image combined with each detections of that image (outer join).
    It also takes into consideration whether the dataframes are empty or not.

    Args:
        ann_df: annotations dataframe
        det_df: detections dataframe

    Return:
        joined_df: The outer join of the two dataframes
    """

    joined_df = ann_df.join(det_df, how="outer", lsuffix="ann", rsuffix="det")
    return joined_df


def compute_iou_det_ann_df(ann_df, det_df, ann_lbl_col, det_lbl_col, iou_tr=0.3):
    """
    Using detections and annotations dataframe compute the polygon IoU between every detection and every markinsg
    for each image.

    Args:
        ann_df: annotations dataframe. Must have at least the following columns: filename, x, y, width, height
        det_df: detections dataframe. Must have at least the following columns: bounds.left, bounds.top, bounds.left,
         bounds.width
         ann_lbl_col: column name for the target label
         det_lbl_col: column name for the predicted label

    Return:
        dictionary with labels as keys and a tuple (joined dataframe with IoU, detection dataframe, anno dataframe) as values
    """
    ann_df = clean_and_set_index_df(ann_df, drop_nan_row_by_columns=[ann_lbl_col])
    det_df = clean_and_set_index_df(det_df)

    res = {}
    for label in set(ann_df[ann_lbl_col].unique()) - set([np.nan]):
        ann_df_label = ann_df[ann_df[ann_lbl_col] == label]
        det_df_label = det_df[det_df[det_lbl_col] == label]

        joined_df = join_dataframes(ann_df_label, det_df_label)
        joined_df["iou"] = joined_df.apply(compute_iou, axis=1)
        joined_df["iou>iou_tr"] = joined_df["iou"] > iou_tr
        res[label] = (joined_df, ann_df_label, det_df_label)

    return res



def compute_stats(joined_with_iou_df, ann_df, det_df):
    """
    Computes the following main statistics: recall (Detection rate), precision, false positive rate(FPR). Also attaches
     to the resulting dictionary the values that were used to compute these values

    Args:
         joined_with_iou_df: pandas dataframe having at least the following columns: image_count, iou>iou_tr
         ann_df: annotations dataframe
         det_df: detections dataframe

    Return:
        dictionary containing the following keys: DR, FPR, image_count, all_markings, all_detections, valid_detections,
        all_fps
    """

    image_count = len(ann_df.index.unique())

    valid_detections = joined_with_iou_df["iou>iou_tr"].sum()
    all_markings = ann_df.shape[0]
    all_detections = det_df.shape[0]

    dr = valid_detections / all_markings if all_markings > 0 else 0

    all_fps = (all_detections - valid_detections)

    fpr = all_fps / image_count if image_count > 0 else 0

    return {"recall/DR": dr, "FPR": fpr, "image_count": image_count, "all_markings": all_markings,
            "all_detections": all_detections, "valid_detections": valid_detections, "all_fps": all_fps}


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


    pass
