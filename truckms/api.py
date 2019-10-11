import numpy as np


class FrameDatapoint:
    """
    Used for passing data from one generator to another in a pipeline
    """
    def __init__(self, image, frame_id):
        assert isinstance(image, np.ndarray)
        self.image = image
        self.frame_id = frame_id


class PredictionDatapoint:
    """
    Encapsulates the output from the neural network
    """

    def __init__(self, pred, frame_id):
        """
        Args:
            pred: dictionary with keys boxes, scores, labels, obj_id ???
            frame_id: id of the frame
        """
        self.pred = pred
        self.frame_id = frame_id

