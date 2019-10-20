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


def gen_cocoitem2datapoints(dataset: CocoDetection):
    """
    CocoDetection dataset returns coordinates as x,y,w,h, but I set the convention to x1,y1,x2,y2
    """
    for i in range(len(dataset)):
        img, gt_target = dataset[i]
        fdp = FrameDatapoint(np.array(img), gt_target[0]['image_id'])

        boxes_xywh = np.array([t['bbox'] for t in gt_target])
        boxes_x1y1x2y2 = boxes_xywh.copy()
        boxes_x1y1x2y2[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
        boxes_x1y1x2y2[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]
        tdp = TargetDatapoint(target={"boxes": boxes_x1y1x2y2.astype(np.int32),
                                      "labels": np.array([coco_id2model_id[t['category_id']] for t in gt_target])},
                              frame_id=gt_target[0]['image_id'])
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
                         'target.label': coco_val_2017_names[label]}
            list_dict_target.append(datapoint)
        if len(target['boxes']) == 0:
            list_dict_target.append({'img_id': frame_id,
                              'target.x1': np.nan, 'target.y1': np.nan, 'target.x2': np.nan, 'target.y2': np.nan,
                              'target.label': np.nan})
    return pd.DataFrame(data=list_dict_pred), pd.DataFrame(data=list_dict_target)



