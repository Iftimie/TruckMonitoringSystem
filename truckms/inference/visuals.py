import matplotlib as mpl
import platform
if platform.system() == "Linux":
    mpl.use('Agg')
import matplotlib.pyplot as plt
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
    def plot_over_image(image, prediction):
        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            x1, y1, x2, y2 = box
            color = PredictionVisualizer.opencv_colors[int(label)]
            cls_name = PredictionVisualizer.model_class_names[int(label)]
            string_to_show = "%s %.2f" % (cls_name, score)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color=color)
            image = cv2.putText(image, string_to_show, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)
        return image