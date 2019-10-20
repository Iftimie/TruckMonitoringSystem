import zipfile
import os
from urllib.request import urlretrieve
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from truckms.api import FrameDatapoint, TargetDatapoint, PredictionDatapoint
import numpy as np
import pandas as pd
from truckms.api import model_class_names, coco_val_2017_names, coco_id2model_id
from typing import Iterable, Tuple
from truckms.inference.visuals import plot_over_image_target


def get_progress_bar_hook():
    pbar = None
    downloaded = 0

    def show_progress(count, block_size, total_size):
        nonlocal pbar
        nonlocal downloaded
        if pbar is None:
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024)
            # pbar = ProgBar(total_size, stream=sys.stdout)

        downloaded += block_size
        pbar.update(block_size)
        if downloaded == total_size:
            pbar.finish()
            pbar = None
            downloaded = 0

    return show_progress

def download_data_if_not_exists(datalake_path):
    if not os.path.exists(datalake_path):
        os.mkdir(datalake_path)
    coco_val_2017_path = os.path.join(datalake_path, "coco_val_2017")
    if not os.path.exists(coco_val_2017_path):
        os.mkdir(coco_val_2017_path)
    if len(os.listdir(coco_val_2017_path)) != 4:
        print("downloading data coco_val_2017")

        zip_path = os.path.join(coco_val_2017_path, "val2017.zip")
        urlretrieve(r"http://images.cocodataset.org/zips/val2017.zip", zip_path, get_progress_bar_hook())
        z = zipfile.ZipFile(zip_path)
        z.extractall(coco_val_2017_path)

        zip_path = os.path.join(coco_val_2017_path, "annotations_trainval2017.zip")
        urlretrieve(r"http://images.cocodataset.org/annotations/annotations_trainval2017.zip", zip_path, get_progress_bar_hook())
        z = zipfile.ZipFile(zip_path)
        z.extractall(coco_val_2017_path)


def get_dataset(datalake_path):
    coco_val_2017_root_path = os.path.join(datalake_path, "coco_val_2017", "val2017")
    coco_val_2017_anno_path = os.path.join(datalake_path, "coco_val_2017", "annotations", "instances_val2017.json")
    dataset = CocoDetection(root=coco_val_2017_root_path, annFile=coco_val_2017_anno_path)
    return dataset


def gen_cocoitem2datapoints(dataset: CocoDetection, frame_ids=None):
    """
    CocoDetection dataset returns coordinates as x,y,w,h, but I set the convention to x1,y1,x2,y2
    """
    id_images_without_targets = len(dataset)
    for i in range(len(dataset)):
        img, gt_target = dataset[i]
        if len(gt_target) != 0:
            if frame_ids is not None and gt_target[0]['image_id'] not in frame_ids:
                continue
            fdp = FrameDatapoint(np.array(img), gt_target[0]['image_id'])

            boxes_xywh = np.array([t['bbox'] for t in gt_target])
            boxes_x1y1x2y2 = boxes_xywh.copy()
            boxes_x1y1x2y2[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
            boxes_x1y1x2y2[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]
            tdp = TargetDatapoint(target={"boxes": boxes_x1y1x2y2.astype(np.int32),
                                          "labels": np.array([coco_id2model_id[t['category_id']] for t in gt_target])},
                                  frame_id=gt_target[0]['image_id'])
            yield fdp, tdp
        else:
            if frame_ids is not None: continue
            fdp = FrameDatapoint(np.array(img), id_images_without_targets)

            boxes_xywh = np.array([t['bbox'] for t in gt_target]).reshape(-1, 4)
            boxes_x1y1x2y2 = boxes_xywh.copy()
            boxes_x1y1x2y2[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
            boxes_x1y1x2y2[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]
            tdp = TargetDatapoint(target={"boxes": boxes_x1y1x2y2.astype(np.int32),
                                          "labels": np.array([coco_id2model_id[t['category_id']] for t in gt_target])},
                                  frame_id=id_images_without_targets)
            id_images_without_targets += 1
            yield fdp, tdp


def gen_cocoitem2targetdp(g: Iterable[Tuple[FrameDatapoint, TargetDatapoint]]):
    for fdp, tdp in g:
        yield tdp


def gen_cocoitem2framedp(g: Iterable[Tuple[FrameDatapoint, TargetDatapoint]]):
    for fdp, tdp in g:
        yield fdp


def target_iter_to_pandas(tdp_iterable: Iterable[TargetDatapoint]):
    """
    Transforms a list or generator of TargetDatapoint into a compact format such as a pandas dataframe.

    Args:
        tdp_iterable: list or generator of TargetDatapoint

    Return:
        pandas dataframe with ground truth values
    """
    list_dict = []
    for tdp in tdp_iterable:
        target = tdp.target
        frame_id = tdp.frame_id
        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = box
            datapoint = {'img_id': frame_id,
                         'target.x1': x1, 'target.y1': y1, 'target.x2': x2, 'target.y2': y2,
                         'target.label': model_class_names[label]}
            list_dict.append(datapoint)
        if len(target['boxes']) == 0:
            list_dict.append({'img_id': frame_id,
                              'target.x1': np.nan, 'target.y1': np.nan, 'target.x2': np.nan, 'target.y2': np.nan,
                              'target.label': np.nan})
    return pd.DataFrame(data=list_dict)


def target_pred_iter_to_pandas(tdp_iterable: Iterable[TargetDatapoint], pdp_iterable: Iterable[PredictionDatapoint]):
    """
    Transforms an iterable of TargetDatapoint and an iterable of PredictionDatapoint to pandas dataframe

    Args:
        tdp_iterable: list or generator of TargetDatapoint
        pdp_iterable: list or generator of PredictionDatapoint

    Return:
        pandas dataframe with prediction
        pandas dataframe with ground truth
    """
    list_dict_pred = []
    list_dict_target = []
    for tdp, pdp in zip(tdp_iterable, pdp_iterable):
        assert pdp.frame_id == tdp.frame_id
        prediction = pdp.pred
        frame_id = pdp.frame_id
        target = tdp.target
        for box, label, score, obj_id in zip(prediction['boxes'], prediction['labels'], prediction['scores'],
                                             prediction['obj_id']):
            x1, y1, x2, y2 = box
            datapoint = {'img_id': frame_id,
                         'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                         'score': score,
                         'label': model_class_names[label],
                         'obj_id': obj_id}
            list_dict_pred.append(datapoint)
        if len(prediction['boxes']) == 0:
            list_dict_pred.append({'img_id': frame_id,
                              'x1': np.nan, 'y1': np.nan, 'x2': np.nan, 'y2': np.nan,
                              'score': np.nan,
                              'label': np.nan,
                              'obj_id': np.nan})
        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = box
            datapoint = {'img_id': frame_id,
                         'target.x1': x1, 'target.y1': y1, 'target.x2': x2, 'target.y2': y2,
                         'target.label': model_class_names[label]}
            list_dict_target.append(datapoint)
        if len(target['boxes']) == 0:
            list_dict_target.append({'img_id': frame_id,
                              'target.x1': np.nan, 'target.y1': np.nan, 'target.x2': np.nan, 'target.y2': np.nan,
                              'target.label': np.nan})
    return pd.DataFrame(data=list_dict_pred), pd.DataFrame(data=list_dict_target)


def plot_targets(fdp_iterable: FrameDatapoint, tdp_iterable: TargetDatapoint) -> FrameDatapoint:
    """
    Plots over the imtages the targets. The number of images should match the number of predictions

    Args:
        fdp_iterable: list of FrameDatapoint or generator
        tdp_iterable: list of TargetDatapoint or generator

    Return:
        generator with FrameDatapoint
    """

    def plots_gen():
        for fdp, pdp in zip(fdp_iterable, tdp_iterable):
            assert fdp.frame_id == pdp.frame_id
            plotted_image = plot_over_image_target(fdp.image, pdp.target)
            yield FrameDatapoint(plotted_image, pdp.frame_id)
    return plots_gen()


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

    precision = valid_detections/all_detections

    all_fps = (all_detections - valid_detections)

    fpr = all_fps / image_count if image_count > 0 else 0

    return {"recall/DR": dr, "precision": precision, "FPR": fpr, "image_count": image_count, "all_markings": all_markings,
            "all_detections": all_detections, "valid_detections": valid_detections, "all_fps": all_fps}
