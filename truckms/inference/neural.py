import torch
import torchvision
from torch.utils.data.dataloader import default_collate
from truckms.inference.visuals import PredictionVisualizer
import numpy as np
import cv2

class TruckDetector:

    def __init__(self, conf_thr=0.5, max_operating_res=320, batch_size=5):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.conf_thr = conf_thr
        self.batch_size = batch_size
        self.max_operating_res = max_operating_res
        self.downsampling_f = None

    def iterable2batch(self, images_iterable):
        batch = []
        image = next(images_iterable)
        m = max(image.shape)
        self.downsampling_f = self.max_operating_res / m
        batch.append(cv2.resize(image, (0, 0), fx=self.downsampling_f, fy=self.downsampling_f))
        for image in images_iterable:
            batch.append(cv2.resize(image, (0, 0), fx=self.downsampling_f, fy=self.downsampling_f))
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
                batch = (default_collate(batch).permute(0, 3, 1, 2).float() / 255.0).to(self.device)
                batch_pred = self.model(batch)
                np_batch_p = [{key: pred[key].cpu().numpy() for key in pred} for pred in batch_pred]
                for pred in np_batch_p:
                    confi_indices = pred['scores'] > self.conf_thr
                    confi_pred = {
                        'boxes': (pred['boxes'][confi_indices] / self.downsampling_f).astype(np.int32),
                        'scores': pred['scores'][confi_indices],
                        'labels': pred['labels'][confi_indices],
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
