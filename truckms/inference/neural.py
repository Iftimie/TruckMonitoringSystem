import torch
import torchvision
from truckms.inference.visuals import PredictionVisualizer
import numpy as np


class TruckDetector:

    def __init__(self, conf_thr=0.5, max_operating_res=800, batch_size=5):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=False, min_size=max_operating_res, max_size=max_operating_res,
                                                                          rpn_pre_nms_top_n_test=100,
                                                                          rpn_post_nms_top_n_test=100,
                                                                          box_score_thresh=conf_thr) #look into class FasterRCNN(GeneralizedRCNN): for more parameters
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.batch_size = batch_size
        self.max_operating_res = max_operating_res
        self.downsampling_f = None

    def iterable2batch(self, images_iterable):
        batch = []
        for idx, image in enumerate(images_iterable):
            batch.append(torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32)).to(self.device) / 255.0)
            if len(batch) == self.batch_size:
                yield batch
                batch.clear()
        yield batch

    def compute(self, images_iterable):
        """
        Computes the predictions for a batch of images received as an iterable. It batch the images internally and work out
        any size mismatches. The images must be in format H, W, C. Images must be in RGB format
        The iterable can be a list or a generator

        #TODO update the compute function for this description.

        Args:
            images_iterable: list of images or generator. it assumes that all generated images have the same resolution
                batching is done internally to avoid memory issues

        Yields:
            list of dicts of boxes, labels, scores:
                for each image in images_iterable, a list with bounding boxes will be returned
        """

        with torch.no_grad():
            for batch in self.iterable2batch(images_iterable):
                batch_pred = self.model(batch)
                np_batch_p = [{key: pred[key].cpu().numpy() for key in pred} for pred in batch_pred]
                for pred in np_batch_p:
                    indices_ttb = pred['labels'] == PredictionVisualizer.model_class_names.index('truck')
                    indices_ttb = np.logical_or(indices_ttb, pred['labels'] == PredictionVisualizer.model_class_names.index('train'))
                    indices_ttb = np.logical_or(indices_ttb, pred['labels'] == PredictionVisualizer.model_class_names.index('bus'))
                    confi_pred = {
                        'boxes': pred['boxes'][indices_ttb],
                        'scores': pred['scores'][indices_ttb],
                        'labels': pred['labels'][indices_ttb],
                    }
                    yield confi_pred

    def plot_detections(self, images_iterable, predictions):
        """
        Plots over the images the detections. The number of images should match the number of predictions

        Args:
            images_iterable: list or generator with images
            predictions: the result of calling .compute()

        Return:
            generator with images with detections
        """

        def plots_gen():
            for image, prediction in zip(images_iterable, predictions):
                yield PredictionVisualizer.plot_over_image(image, prediction)

        return plots_gen()
