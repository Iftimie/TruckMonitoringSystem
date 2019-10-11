import matplotlib as mpl
import platform
if platform.system() == "Linux":
    mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
from truckms.inference.neural import model_class_names


colormap = plt.get_cmap('gist_rainbow')

colors = []
for i in range(len(model_class_names)):
    colors.append(colormap(1. * i / len(model_class_names)) )
# colors = [for i in range(len(model_class_names))]
opencv_colors = [[i * 255 for i in color] for color in colors]


def plot_over_image(image, prediction):
    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        x1, y1, x2, y2 = box
        color = opencv_colors[int(label)]
        cls_name = model_class_names[int(label)]
        string_to_show = "%s %.2f" % (cls_name, score)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color=color)
        image = cv2.putText(image, string_to_show, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)
    return image
