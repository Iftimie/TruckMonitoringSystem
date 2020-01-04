import matplotlib as mpl
import platform
if platform.system() == "Linux":
    mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
import math
from truckms.api import model_class_names


colormap = plt.get_cmap('gist_rainbow')

colors = []
for i in range(len(model_class_names)):
    colors.append(colormap(1. * i / len(model_class_names)) )
# colors = [for i in range(len(model_class_names))]
opencv_colors = [[i * 255 for i in color] for color in colors]


def plot_over_image(image, pred, index=None):
    """
    Plots the predicted bounding boxes onto an image

    Args:
        image: np.array
        d: dictionary with keys boxes, labels, scores, obj_id

    Return:
        plotted image with bounding boxes and a string over the bounding box containing the class name, confidence score
        and object id (from tracking)
    """
    image = image.copy()
    for box, label, score, obj_id in zip(pred['boxes'], pred['labels'], pred['scores'], pred['obj_id']):
        obj_id = obj_id if not math.isnan(obj_id) else -1
        x1, y1, x2, y2 = box
        color = opencv_colors[int(label)]
        cls_name = model_class_names[int(label)]
        string_to_show = "%s %.2f %d" % (cls_name, score, obj_id)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color=color)
        image = cv2.putText(image, string_to_show, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)
    if index is not None:
        image = cv2.putText(image, "index {}".format(index), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
    return image


def plot_over_image_target(image, target):
    """
    Plots the predicted bounding boxes onto an image

    Args:
        image: np.array
        d: dictionary with keys boxes, labels, scores, obj_id

    Return:
        plotted image with bounding boxes and a string over the bounding box containing the class name, confidence score
        and object id (from tracking)
    """
    image = image.copy()
    for box, label in zip(target['boxes'], target['labels']):
        x1, y1, x2, y2 = box
        color = opencv_colors[int(label)]
        cls_name = model_class_names[int(label)]
        string_to_show = "%s" % (cls_name,)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color=color)
        image = cv2.putText(image, string_to_show, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)

    return image

