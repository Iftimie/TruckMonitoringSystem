from truckms.inference.analytics import Tracker
import numpy as np
import cv2
import os.path as osp
from truckms.inference.neural import TruckDetector
import pandas as pd
from truckms.inference.analytics import pipeline
import os
"""
credits to https://github.com/kcg2015/Vehicle-Detection-and-Tracking/
"""

def draw_box_label(img, bbox_cv2, box_color=(0, 255, 255), show_label=True):
    '''
    Helper funciton for drawing the bounding boxes and the labels
    bbox_cv2 = [left, top, right, bottom]
    '''
    # box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]

    # Draw the bounding box
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 4)

    if show_label:
        # Draw a filled box on top of the bounding box (as the background for the labels)
        cv2.rectangle(img, (left - 2, top - 45), (right + 2, top), box_color, -1, 1)

        # Output the labels that show the x and y coordinates of the bounding box center.
        text_x = 'x=' + str((left + right) / 2)
        cv2.putText(img, text_x, (left, top - 25), font, font_size, font_color, 1, cv2.LINE_AA)
        text_y = 'y=' + str((top + bottom) / 2)
        cv2.putText(img, text_y, (left, top - 5), font, font_size, font_color, 1, cv2.LINE_AA)

    return img

def test_tracker():
    import matplotlib.pyplot as plt
    import glob

    # Creat an instance
    trk = Tracker()
    # Test R_ratio
    trk.R_scaler = 1.0 / 16
    # Update measurement noise covariance matrix
    trk.update_R()
    # Initial state
    x_init = np.array([390, 0, 1050, 0, 513, 0, 1278, 0])
    x_init_box = [x_init[0], x_init[2], x_init[4], x_init[6]]
    # Measurement
    z = np.array([399, 1022, 504, 1256])
    trk.x_state = x_init.T
    trk.kalman_filter(z.T)
    # Updated state
    x_update = trk.x_state
    x_updated_box = [x_update[0], x_update[2], x_update[4], x_update[6]]

    print('The initial state is: ', x_init)
    print('The measurement is: ', z)
    print('The update state is: ', x_update)

    # Visualize the Kalman filter process and the
    # impact of measurement nosie convariance matrix

    test_image = osp.join(osp.dirname(__file__), 'data', 'test_image.PNG')
    img = cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 14))
    draw_box_label(img, x_init_box, box_color=(0, 255, 0))
    ax = plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.title('Initial: ' + str(x_init_box))

    draw_box_label(img, z, box_color=(255, 0, 0))
    ax = plt.subplot(3, 1, 2)
    plt.imshow(img)
    plt.title('Measurement: ' + str(z))

    draw_box_label(img, x_updated_box)
    ax = plt.subplot(3, 1, 3)
    plt.imshow(img)
    plt.title('Updated: ' + str(x_updated_box))
    plt.show()


def test_pipeline():
    csv_file_path = osp.join(osp.dirname(__file__), 'data', 'cut.csv')
    pred_gen_from_df = TruckDetector.pandas_to_pred_iter(pd.read_csv(csv_file_path))
    tracker_list = []
    for idx, (pred, img_id) in enumerate(pred_gen_from_df):
        z_box = pred['boxes'].tolist()
        good_boxes, tracker_list = pipeline(z_box, tracker_list)

        print (len(z_box))
        print (len(good_boxes))
        if idx == 10:break
