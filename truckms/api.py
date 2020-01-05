import numpy as np


model_class_names = ["__background__ ", 'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
                     'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                     'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                     'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                     'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                     'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'potted plant', 'bed',
                     'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                     'teddy bear', 'hair drier', 'toothbrush']
# these are from model prediction

coco_val_2017_names = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorbike', 5: 'aeroplane', 6: 'bus', 7: 'train',
                       8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign',
                       14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
                       21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                       28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis',
                       36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
                       41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass',
                       47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
                       54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
                       60: 'donut', 61: 'cake', 62: 'chair', 63: 'sofa', 64: 'potted plant', 65: 'bed',
                       67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
                       76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                       82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
                       89: 'hair drier', 90: 'toothbrush'}
coco_id2model_id = {k: model_class_names.index(coco_val_2017_names[k]) for k in coco_val_2017_names}
# these are from annotations


class FrameDatapoint:
    """
    Used for passing data from one generator to another in a pipeline

    The image must have an id_. id_ can be set to anything or to None, however it will help
             identify the position in movie if frames are skipped

    The image should be in RGB format. It the image was read from disk with opencv then it should be converted from BGR to RGB
    """
    def __init__(self, image, frame_id, reason=None):
        assert isinstance(image, np.ndarray)
        self.image = image
        self.frame_id = frame_id
        self.reason = reason

    def __repr__(self):
        return "frame_id{}".format(self.frame_id)


class BatchedFrameDatapoint:
    """
    Used for passing data from one generator to another in a pipeline when using PyTorch framework

    The images must have an id_. id_ can be set to a list of integers or to a list of None, however it will help
             identify the position in movie if frames are skipped if it is an integer

    #TODO do I really need this?
    """
    def __init__(self, batch_images, batch_frames_ids, batch_reason):
        assert isinstance(batch_images, list)
        assert isinstance(batch_frames_ids, list)
        assert isinstance(batch_reason, list)
        if len(batch_images)>0:
            assert len(batch_images[0]) == 3
        assert len(batch_frames_ids) == len(batch_images) == len(batch_reason)
        self.batch_images = batch_images
        self.batch_frames_ids = batch_frames_ids
        self.batch_reason = batch_reason


class PredictionDatapoint:
    """
    Encapsulates the output from the neural network
    """

    def __init__(self, pred, frame_id, reason=None):
        """
        Args:
            pred: dictionary with keys boxes, scores, labels, obj_id
            frame_id: id of the frame
            reason: string describing the reason for which this prediction datapoint is created. for example it could be
                None if there is no particular reason, or a string such as "movement" meaning that at this frame_id, there was a motion detected,
                although there was nothing detected
        """
        array_keys = ['boxes', 'scores', 'labels', 'obj_id']
        assert all(isinstance(pred[k], np.ndarray) for k in array_keys)
        assert len(pred['boxes'].shape) == 2
        assert len(pred['scores'].shape) == 1
        assert len(pred['labels'].shape) == 1
        assert len(pred['obj_id'].shape) == 1
        assert pred['obj_id'].shape[0] == pred['labels'].shape[0]
        assert pred['scores'].shape[0] == pred['boxes'].shape[0]
        assert pred['scores'].shape[0] == pred['labels'].shape[0]
        self.pred = pred
        self.frame_id = frame_id
        self.reason = reason

    def __repr__(self):
        return "frame_id{}".format(self.frame_id)


class TargetDatapoint:
    """
    Encapsulated the ground truth for an image
    """

    def __init__(self, target, frame_id):
        """
        Args:
            target: dictionary with keys "boxes", "labels"
            frame_id: id of the frame
        """
        assert len(target['boxes'].shape) == 2
        assert len(target['labels'].shape) == 1
        self.target = target
        self.frame_id = frame_id

