import matplotlib as mpl
import platform
if platform.system() == "Linux":
    mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image
import numpy as np
from deprecated import deprecated
import cv2

class PredictionVisualizer:

    colormap = plt.get_cmap('gist_rainbow')
    model_class_names = ["__background__ ", 'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
                         'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                         'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                         'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                         'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                         'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet',
                         'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                         'teddy bear', 'hair drier', 'toothbrush']
    colors = []
    for i in range(len(model_class_names)):
        colors.append(colormap(1. * i / len(model_class_names)) )
    # colors = [for i in range(len(model_class_names))]
    opencv_colors = [[i * 255 for i in color] for color in colors]



    @staticmethod
    @deprecated(reason="matplotlib is slow")
    def plot_over_image_plt(image, prediction):
        dpi = 100
        fig = plt.figure()
        fig.set_size_inches(image.shape[1] / dpi, image.shape[0] / dpi)
        ax = plt.gca()
        ax.imshow(image)
        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            x1, y1, x2, y2 = box
            box_h, box_w = y2 - y1, x2 - x1
            color = PredictionVisualizer.colors[int(label)]
            cls_name = PredictionVisualizer.model_class_names[int(label)]
            string_to_show = "%s %.2f" % (cls_name, score)
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(bbox)
            plt.text(x1, y1, s=string_to_show, color='white', verticalalignment='top', bbox={'color': color, 'pad': 0})

        fig.canvas.draw()
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches='tight', pad_inches=0.0)
        buf.seek(0)
        im = Image.open(buf)
        image = np.array(im)

        return image

    @staticmethod
    def plot_over_image(image, prediction):
        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            x1, y1, x2, y2 = box
            color = PredictionVisualizer.opencv_colors[int(label)]
            cls_name = PredictionVisualizer.model_class_names[int(label)]
            string_to_show = "%s %.2f" % (cls_name, score)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color=color)
            image = cv2.putText(image, string_to_show, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)
        return image