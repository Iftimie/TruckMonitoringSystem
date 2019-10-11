import matplotlib as mpl
import platform
if platform.system() == "Linux":
    mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2


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


colormap = plt.get_cmap('gist_rainbow')

colors = []
for i in range(len(model_class_names)):
    colors.append(colormap(1. * i / len(model_class_names)) )
# colors = [for i in range(len(model_class_names))]
opencv_colors = [[i * 255 for i in color] for color in colors]


def plot_over_image(image, pred):
    """
    Plots the predicted bounding boxes onto an image

    Args:
        image: np.array
        pred: dictionary with keys boxes, labels, scores, obj_id

    Return:
        plotted image with bounding boxes and a string over the bounding box containing the class name, confidence score
        and object id (from tracking)
    """
    for box, label, score, obj_id in zip(pred['boxes'], pred['labels'], pred['scores'], pred['obj_id']):
        obj_id = obj_id if obj_id is not None else -1
        x1, y1, x2, y2 = box
        color = opencv_colors[int(label)]
        cls_name = model_class_names[int(label)]
        string_to_show = "%s %.2f %d" % (cls_name, score, obj_id)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color=color)
        image = cv2.putText(image, string_to_show, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)
    return image
