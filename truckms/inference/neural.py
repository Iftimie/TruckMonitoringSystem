import torch
import torchvision
from truckms.inference.visuals import PredictionVisualizer
import numpy as np
import pandas as pd
import math

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
        batch_ids = []
        for idx, (image, id_) in enumerate(images_iterable):
            batch.append(torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32)).to(self.device) / 255.0)
            batch_ids.append(id_)
            if len(batch) == self.batch_size:
                yield batch, batch_ids
                batch.clear()
                batch_ids.clear()
        yield batch, batch_ids

    def compute(self, images_iterable):
        """
        Computes the predictions for a batch of images received as an iterable. It batch the images internally and work out
        any size mismatches. The images must be in format H, W, C. Images must be in RGB format
        The iterable can be a list or a generator

        #TODO update the compute function for this description.

        Args:
            images_iterable: list of tuples (images, id_) or generator. it assumes that all generated images have the same resolution
                batching is done internally to avoid memory issues
                the image must have an id_. can be set to anything or to None, however it will help identify the position in movie if frames are skipped

        Yields:
            dicts of boxes, labels, scores:
            and the id_ for the image
                for each image in images_iterable, a list with bounding boxes will be returned
        """

        with torch.no_grad():
            for batch, batch_ids in self.iterable2batch(images_iterable):
                if len(batch) !=0:
                    batch_pred = self.model(batch)
                    np_batch_p = [{key: pred[key].cpu().numpy() for key in pred} for pred in batch_pred]
                    for pred, id_ in zip(np_batch_p, batch_ids):
                        indices_ttb = pred['labels'] == PredictionVisualizer.model_class_names.index('truck')
                        indices_ttb = np.logical_or(indices_ttb, pred['labels'] == PredictionVisualizer.model_class_names.index('train'))
                        indices_ttb = np.logical_or(indices_ttb, pred['labels'] == PredictionVisualizer.model_class_names.index('bus'))
                        indices_ttb = np.logical_or(indices_ttb, pred['labels'] == PredictionVisualizer.model_class_names.index('car'))
                        confi_pred = {
                            'boxes': pred['boxes'][indices_ttb].astype(np.int32),
                            'scores': pred['scores'][indices_ttb],
                            'labels': pred['labels'][indices_ttb],
                            'obj_id': [None] * np.sum(indices_ttb)
                        }
                        yield confi_pred, id_

    @staticmethod
    def pred_iter_to_pandas(predictions_iterable):
        list_dict = []
        for prediction, id_ in predictions_iterable:
            for box, label, score, obj_id in zip(prediction['boxes'], prediction['labels'], prediction['scores'], prediction['obj_id']):
                x1, y1, x2, y2 = box
                datapoint = {'img_id': id_,
                             'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                             'score': score,
                             'label':PredictionVisualizer.model_class_names[label],
                             'obj_id':obj_id}
                list_dict.append(datapoint)
            if len(prediction['boxes']) == 0:
                list_dict.append({'img_id': id_,
                                  'x1': None, 'y1': None, 'x2': None, 'y2': None,
                                  'score': None,
                                  'label': None,
                                  'obj_id': None})
        return pd.DataFrame(data=list_dict)

    @staticmethod
    def pandas_to_pred_iter(data_frame):
        """
        generator that yields detections for each frame and the frame id from a pandas dataframe
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
            if not any(datapoint[key] is None for key in datapoint) and not any(math.isnan(datapoint[k]) for k in datapoint if not isinstance(datapoint[k], str)):
                list_boxes.append([datapoint['x1'], datapoint['y1'], datapoint['x2'], datapoint['y2']])
                list_scores.append(datapoint['score'])
                list_labels.append(PredictionVisualizer.model_class_names.index(datapoint['label']))
                list_obj_id.append(datapoint['obj_id'])

        prediction = {'boxes': np.array(list_boxes).astype(np.int32),
                      'scores': np.array(list_scores),
                      'labels': np.array(list_labels),
                      'obj_id': np.array(list_obj_id)}
        yield prediction, id_

    @staticmethod
    def plot_detections(images_iterable, predictions_iterable):
        """
        Plots over the images the detections. The number of images should match the number of predictions

        Args:
            images_iterable: list of tuples (images, id_) or generator
            predictions_iterable: the result of calling .compute()

        Return:
            generator with images with detections
        """

        def plots_gen():
            for (image, id_img), (prediction, id_pred) in zip(images_iterable, predictions_iterable):
                print (id_img, id_pred)
                assert id_img == id_pred
                yield PredictionVisualizer.plot_over_image(image, prediction), id_img

        return plots_gen()


