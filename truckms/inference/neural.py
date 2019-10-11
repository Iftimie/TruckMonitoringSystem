from truckms.inference.visuals import plot_over_image
from truckms.inference.utils import batch_image_id_yielder, prediction_id_yielder
import numpy as np
import pandas as pd
import torchvision
from functools import reduce
import torch
import math


device = 'cuda' if torch.cuda.is_available() else "cpu"

model_class_names = ["__background__ ", 'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
                     'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                     'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                     'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                     'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                     'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
                     'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                     'teddy bear', 'hair drier', 'toothbrush']


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


# TODO do the following lines. the decorators should actually check this
# @yielder(image_id_tuple)
# @inputs({"images_ids_iterable": image_id_tuple})
@batch_image_id_yielder
def iterable2batch(images_ids_iterable, batch_size=5):
    """
    Transforms an iterable of images and ids into batches of images and ids

    Args:
        images_ids_iterable: list, or generator that yields images and their frame ids
        batch_size: the size of the yielded batch

    Yields:
        batch of images as tensor of shape [batch_size, C, H, W]
        batch of ids as list
    """
    batch = []
    batch_ids = []
    for idx, (image, id_) in enumerate(images_ids_iterable):
        batch.append(torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32)).to(device) / 255.0)
        batch_ids.append(id_)
        if len(batch) == batch_size:
            yield batch, batch_ids
            batch.clear()
            batch_ids.clear()
    yield batch, batch_ids


@prediction_id_yielder
# @yielder(pred_id_tuple) # it does not output a list
# @inputs({"images_ids_iterable": image_id_tuple})
def compute(images_ids_iterable, model, filter_classes=None, ):
    """
    Computes the predictions for a batch of images received as an iterable. It batches the images internally and works
     out any size mismatches. The images must be in format H, W, C. Images must be in RGB format.
    The iterable can be a list or a generator

    Args:
        images_ids_iterable: list of tuples (images, id_) or generator. it assumes that all generated images have the
            same resolution. batching is done internally to avoid memory issues. The image must have an id_. id_ can be
            set to anything or to None, however it will help identify the position in movie if frames are skipped
        model: Faster-RCNN model
        filter_classes: only classes found in this argument will be yielded

    Yields:
        dict having keys boxes, labels, scores and obj_id
        id_ for the image
    """
    if filter_classes is None:
        filter_classes = ['truck', 'train', 'bus', 'car']

    with torch.no_grad():
        for batch, batch_ids in iterable2batch(images_ids_iterable):
            if len(batch) == 0:
                break

            batch_pred = [{key: pred[key].cpu().numpy() for key in pred} for pred in model(batch)]  # tensor to numpy
            for pred, id_ in zip(batch_pred, batch_ids):
                valid_inds = reduce(np.logical_or,
                                    (pred['labels'] == model_class_names.index(lbl) for lbl in filter_classes),
                                    np.ones(pred['labels'].shape, dtype=bool))
                to_yield_pred = {
                    'boxes': pred['boxes'][valid_inds].astype(np.int32),
                    'scores': pred['scores'][valid_inds],
                    'labels': pred['labels'][valid_inds],
                    'obj_id': [None] * np.sum(valid_inds)
                }
                yield to_yield_pred, id_


# @inputs({"predictions_iterable": pred_id_tuple})
def pred_iter_to_pandas(pred_id_iterable):
    """
    Transforms a list or generator of predictions with frame ids into a compact format such as a pandas dataframe.

    Args:
        pred_id_iterable: list or generator of predictions with frame ids

    Return:
        pandas dataframe of detections
    """
    list_dict = []
    for prediction, id_ in pred_id_iterable:
        for box, label, score, obj_id in zip(prediction['boxes'], prediction['labels'], prediction['scores'],
                                             prediction['obj_id']):
            x1, y1, x2, y2 = box
            datapoint = {'img_id': id_,
                         'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                         'score': score,
                         'label': model_class_names[label],
                         'obj_id': obj_id}
            list_dict.append(datapoint)
        if len(prediction['boxes']) == 0:
            list_dict.append({'img_id': id_,
                              'x1': None, 'y1': None, 'x2': None, 'y2': None,
                              'score': None,
                              'label': None,
                              'obj_id': None})
    return pd.DataFrame(data=list_dict)


# @yielder({"predictions_iterable": pred_id_tuple})
def pandas_to_pred_iter(data_frame):
    """
    Generator that yields detections for each frame and the frame id from a pandas dataframe

    Args:
        data_frame: pandas dataframe with detections

    Yields:
        dict having keys boxes, labels, scores and obj_id
        id_ for the image
    """
    list_dict = list(data_frame.T.to_dict().values())

    list_boxes, list_scores, list_labels, list_obj_id = [], [], [], []
    id_ = list_dict[0]['img_id']

    for datapoint in list_dict:
        img_id = datapoint['img_id']
        if img_id != id_:
            prediction = {'boxes': np.array(list_boxes).astype(np.int32),
                          'scores': np.array(list_scores),
                          'labels': np.array(list_labels),
                          'obj_id': np.array(list_obj_id)}
            yield prediction, id_
            id_ = img_id
            list_boxes, list_scores, list_labels = [], [], []
        if not any(datapoint[key] is None for key in datapoint if key != 'obj_id')\
                and not any(math.isnan(datapoint[k]) for k in datapoint if k != 'obj_id' and not isinstance(datapoint[k], str)):
            list_boxes.append([datapoint['x1'], datapoint['y1'], datapoint['x2'], datapoint['y2']])
            list_scores.append(datapoint['score'])
            list_labels.append(model_class_names.index(datapoint['label']))
            list_obj_id.append(datapoint['obj_id'])

    prediction = {'boxes': np.array(list_boxes).astype(np.int32),
                  'scores': np.array(list_scores),
                  'labels': np.array(list_labels),
                  'obj_id': np.array(list_obj_id)}
    yield prediction, id_


# @inputs({"image_id_iterable": image_id_tuple, "pred_id_iterable": pred_id_tuple})
def plot_detections(image_id_iterable, pred_id_iterable):
    """
    Plots over the imtages the deections. The number of images should match the number of predictions

    Args:
        image_id_iterable: list of tuples (images, id_) or generator
        pred_id_iterable: list of tuples (prediction, id_) or generator
            for example it can be the result of calling .compute() or pandas_to_pred_iter()

    Return:
        generator with images with detections
    """

    def plots_gen():
        for (image, id_img), (prediction, id_pred) in zip(image_id_iterable, pred_id_iterable):
            assert id_img == id_pred
            yield plot_over_image(image, prediction), id_img

    return plots_gen()
