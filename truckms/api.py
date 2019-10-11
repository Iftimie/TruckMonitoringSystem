import numpy as np


class FrameDatapoint:
    """
    Used for passing data from one generator to another in a pipeline

    The image must have an id_. id_ can be set to anything or to None, however it will help
             identify the position in movie if frames are skipped
    """
    def __init__(self, image, frame_id):
        assert isinstance(image, np.ndarray)
        self.image = image
        self.frame_id = frame_id


class BatchedFrameDatapoint:
    """
    Used for passing data from one generator to another in a pipeline when using PyTorch framework

    The images must have an id_. id_ can be set to a list of integers or to a list of None, however it will help
             identify the position in movie if frames are skipped if it is an integer
    """
    def __init__(self, batch_images, batch_frames_ids):
        assert isinstance(batch_images, list)
        assert isinstance(batch_frames_ids, list)
        if len(batch_images)>0:
            assert len(batch_images[0]) == 3
        assert len(batch_frames_ids) == len(batch_images)
        self.batch_images = batch_images
        self.batch_frames_ids = batch_frames_ids


class PredictionDatapoint:
    """
    Encapsulates the output from the neural network
    """

    def __init__(self, pred, frame_id):
        """
        Args:
            pred: dictionary with keys boxes, scores, labels, obj_id
            frame_id: id of the frame
        """
        assert all(isinstance(pred[k], np.ndarray) for k in pred)
        assert len(pred['boxes'].shape) == 2
        assert len(pred['scores'].shape) == 1
        assert len(pred['labels'].shape) == 1
        assert len(pred['obj_id'].shape) == 1
        assert pred['obj_id'].shape[0] == pred['labels'].shape[0]
        assert pred['scores'].shape[0] == pred['boxes'].shape[0]
        assert pred['scores'].shape[0] == pred['labels'].shape[0]
        self.pred = pred
        self.frame_id = frame_id

