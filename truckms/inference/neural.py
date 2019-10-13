from truckms.inference.visuals import plot_over_image, model_class_names
from truckms.api import FrameDatapoint, BatchedFrameDatapoint, PredictionDatapoint
from functools import reduce
import numpy as np
import pandas as pd
import torchvision
import torch
import math


device = 'cuda' if torch.cuda.is_available() else "cpu"


def create_model(conf_thr=0.5, max_operating_res=800):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=False,
                                                                 min_size=max_operating_res,
                                                                 max_size=max_operating_res,
                                                                 rpn_pre_nms_top_n_test=100,
                                                                 rpn_post_nms_top_n_test=100,
                                                                 box_score_thresh=conf_thr)
    # look into class FasterRCNN(GeneralizedRCNN): for more parameters
    model.eval()
    model = model.to(device)
    return model


def iterable2batch(fdp_iterable: FrameDatapoint, batch_size=5, cdevice=device):
    """
    Transforms an iterable of FrameDatapoint into BatchedFrameDatapoint

    Args:
        fdp_iterable: list, or generator that yields FrameDatapoint
        batch_size: the size of the yielded batch

    Yields:
        BatchedFrameDatapoint
    """
    batch = []
    batch_ids = []
    for idx, fdp in enumerate(fdp_iterable):
        image = fdp.image
        id_ = fdp.frame_id
        batch.append(torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32)).to(cdevice) / 255.0)
        batch_ids.append(id_)
        if len(batch) == batch_size:
            yield BatchedFrameDatapoint(batch, batch_ids)
            batch.clear()
            batch_ids.clear()
    yield BatchedFrameDatapoint(batch, batch_ids)


def compute(fdp_iterable: FrameDatapoint, model, filter_classes=None, batch_size=5, cdevice=device) -> PredictionDatapoint:
    """
    Computes the predictions for an iterable of FrameDatapoint. It batches the images internally and works. The images
    must be in format H, W, C. Images must be in RGB format.

    Args:
        fdp_iterable: list of FrameDatapoint or generator. it assumes that all generated images have the
            same resolution.
        model: Faster-RCNN model
        filter_classes: only classes found in this argument will be yielded
        batch_size: number of images to forward at once

    Yields:
        PredictionDatapoint
    """
    if filter_classes is None:
        filter_classes = ['truck', 'train', 'bus', 'car']

    with torch.no_grad():
        for bfdp in iterable2batch(fdp_iterable, batch_size=batch_size, cdevice=cdevice):
            if len(bfdp.batch_images) == 0:
                break

            # tensor to numpy
            batch_pred = [{key: pred[key].cpu().numpy() for key in pred} for pred in model(bfdp.batch_images)]
            for pred, frame_id in zip(batch_pred, bfdp.batch_frames_ids):
                valid_inds = reduce(np.logical_or,
                                    (pred['labels'] == model_class_names.index(lbl) for lbl in filter_classes),
                                    np.zeros(pred['labels'].shape, dtype=bool))
                to_yield_pred = {
                    'boxes': pred['boxes'][valid_inds].astype(np.int32),
                    'scores': pred['scores'][valid_inds],
                    'labels': pred['labels'][valid_inds],
                    'obj_id': np.array([np.nan] * np.sum(valid_inds))
                }
                yield PredictionDatapoint(to_yield_pred, frame_id)


def pred_iter_to_pandas(pdp_iterable):
    """
    Transforms a list or generator of PredictionDatapoint into a compact format such as a pandas dataframe.

    Args:
        pdp_iterable: list or generator of PredictionDatapoint

    Return:
        pandas dataframe with detections
    """
    list_dict = []
    for pdp in pdp_iterable:
        prediction = pdp.pred
        frame_id = pdp.frame_id
        for box, label, score, obj_id in zip(prediction['boxes'], prediction['labels'], prediction['scores'],
                                             prediction['obj_id']):
            x1, y1, x2, y2 = box
            datapoint = {'img_id': frame_id,
                         'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                         'score': score,
                         'label': model_class_names[label],
                         'obj_id': obj_id}
            list_dict.append(datapoint)
        if len(prediction['boxes']) == 0:
            list_dict.append({'img_id': frame_id,
                              'x1': np.nan, 'y1': np.nan, 'x2': np.nan, 'y2': np.nan,
                              'score': np.nan,
                              'label': np.nan,
                              'obj_id': np.nan})
    return pd.DataFrame(data=list_dict)


def pandas_to_pred_iter(data_frame) -> PredictionDatapoint:
    """
    Generator that yields PredictionDatapoint from a pandas dataframe

    Args:
        data_frame: pandas dataframe with detections

    Yields:
        PredictionDatapoint
    """
    list_dict = list(data_frame.T.to_dict().values())

    list_boxes, list_scores, list_labels, list_obj_id = [], [], [], []
    frame_id = list_dict[0]['img_id']

    for datapoint in list_dict:
        img_id = datapoint['img_id']
        if img_id != frame_id:
            prediction = {'boxes': np.array(list_boxes).astype(np.int32).reshape(-1, 4),
                          'scores': np.array(list_scores),
                          'labels': np.array(list_labels),
                          'obj_id': np.array(list_obj_id)}
            yield PredictionDatapoint(prediction, frame_id)
            frame_id = img_id
            list_boxes, list_scores, list_labels, list_obj_id = [], [], [], []
        if not any(math.isnan(datapoint[key]) for key in datapoint if key != 'label')\
                and not any(math.isnan(datapoint[k]) for k in datapoint if k != 'obj_id' and not isinstance(datapoint[k], str)):
            list_boxes.append([datapoint['x1'], datapoint['y1'], datapoint['x2'], datapoint['y2']])
            list_scores.append(datapoint['score'])
            list_labels.append(model_class_names.index(datapoint['label']))
            list_obj_id.append(datapoint['obj_id'])

    prediction = {'boxes': np.array(list_boxes).astype(np.int32).reshape(-1, 4),
                  'scores': np.array(list_scores),
                  'labels': np.array(list_labels),
                  'obj_id': np.array(list_obj_id)}
    yield PredictionDatapoint(prediction, frame_id)


def plot_detections(fdp_iterable: FrameDatapoint, pdp_iterable: PredictionDatapoint) -> FrameDatapoint:
    """
    Plots over the imtages the deections. The number of images should match the number of predictions

    Args:
        fdp_iterable: list of FrameDatapoint or generator
        pdp_iterable: list of PredictionDatapoint or generator
            for example it can be the result of calling .compute() or pandas_to_pred_iter()

    Return:
        generator with FrameDatapoint
    """

    def plots_gen():
        for fdp, pdp in zip(fdp_iterable, pdp_iterable):
            assert fdp.frame_id == pdp.frame_id
            plotted_image = plot_over_image(fdp.image, pdp.pred)
            yield FrameDatapoint(plotted_image, pdp.frame_id)

    return plots_gen()


