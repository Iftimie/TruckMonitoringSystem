import numpy as np
from scipy.optimize import linear_sum_assignment
"Credits to https://github.com/kcg2015/Vehicle-Detection-and-Tracking/blob/master/"

# !/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Implement and test tracker
'''
import numpy as np
from numpy import dot
from scipy.linalg import inv, block_diag
from truckms.api import PredictionDatapoint
from typing import Iterable


class Tracker():  # class for Kalman Filter-based tracker
    def __init__(self):
        # Initialize parametes for tracker (history)
        self.id = 0  # tracker's id
        self.box = []  # list to store the coordinates for a bounding box
        self.score = None
        self.label = None
        self.hits = 0  # number of detection matches
        self.no_losses = 0  # number of unmatched tracks (track loss)
        # TODO maybe add the class of the tracked bbox

        # Initialize parameters for Kalman Filtering
        # The state is the (x, y) coordinates of the detection box
        # state: [up, up_dot, left, left_dot, down, down_dot, right, right_dot]
        # or[up, up_dot, left, left_dot, height, height_dot, width, width_dot]
        self.x_state = []
        self.dt = 1.  # time interval

        # Process matrix, assuming constant velocity model
        self.F = np.array([[1, self.dt, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, self.dt, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, self.dt, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, self.dt],
                           [0, 0, 0, 0, 0, 0, 0, 1]])

        # Measurement matrix, assuming we can only measure the coordinates

        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0]])

        # Initialize the state covariance
        self.L = 10.0
        self.P = np.diag(self.L * np.ones(8))

        # Initialize the process covariance
        self.Q_comp_mat = np.array([[self.dt ** 4 / 4., self.dt ** 3 / 2.],
                                    [self.dt ** 3 / 2., self.dt ** 2]])
        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat,
                            self.Q_comp_mat, self.Q_comp_mat)

        # Initialize the measurement covariance
        self.R_scaler = 1.0
        self.R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(self.R_diag_array)

    def update_R(self):
        R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(R_diag_array)

    def kalman_filter(self, z):
        '''
        Implement the Kalman Filter, including the predict and the update stages,
        with the measurement z
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        # Update
        S = dot(self.H, self.P).dot(self.H.T) + self.R
        K = dot(self.P, self.H.T).dot(inv(S))  # Kalman gain
        y = z - dot(self.H, x)  # residual
        x += dot(K, y)
        self.P = self.P - dot(K, self.H).dot(self.P)
        self.x_state = x.astype(int)  # convert to integer coordinates
        # (pixel values)

    def predict_only(self):
        '''
        Implment only the predict stage. This is used for unmatched detections and
        unmatched tracks
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q
        self.x_state = x.astype(int)


def box_iou(a, b):
    '''
    Helper funciton to calculate the ratio between intersection and the union of
    two boxes a and b
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    '''

    w_intsec = np.maximum(0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum(0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0]) * (a[3] - a[1])
    s_b = (b[2] - b[0]) * (b[3] - b[1])

    return float(s_intsec) / (s_a + s_b - s_intsec)


def assign_detections_to_trackers(trackers, detections, iou_thrd=0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''

    IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for t, trk in enumerate(trackers):
        # trk = convert_to_cv2bbox(trk)
        for d, det in enumerate(detections):
            #   det = convert_to_cv2bbox(det)
            IOU_mat[t, d] = box_iou(trk, det)
            # TODO maybe compute the IOU for separate classes

            # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    indices = linear_sum_assignment(-IOU_mat)
    indices = np.asarray(indices)
    matched_idx = np.transpose(indices)

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if (t not in matched_idx[:, 0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if (d not in matched_idx[:, 1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object

    for m in matched_idx:
        if (IOU_mat[m[0], m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Detection:
    def __init__(self, box, lablel, score, id=None):
        self.box = box # x1, y1, x2, y2
        self.label = lablel
        self.score = score
        self.id = id


def detections_list2_dict_numpy(all_boxes):
    boxes, scores, labels, ids_ = [], [], [], []
    for item in all_boxes:
        boxes.append(item.box)
        scores.append(item.score)
        labels.append(item.label)
        ids_.append(item.id)
    pred = {"boxes": np.array(boxes).astype(np.int32).reshape(-1, 4),
            'labels': np.array(labels),
            'scores': np.array(scores),
            'obj_id': np.array(ids_)}
    return pred


def dict_numpy2detections_list(pred):
    detections = [Detection(pred['boxes'][i], pred['labels'][i], pred['scores'][i]) for i in range(len(pred['boxes']))]
    return detections


def pipeline(pred, tracker_list, id_incrementer, max_age=4, min_hits=2):
    """
        Pipeline function for detection and tracking

        Args:
            pred: output of neural network API. dictionary with keys boxes, scores, labels and obj_id
            max_age: no.of consecutive unmatched detection before a track is deleted
            min_hits: no. of consecutive matches needed to establish a track

    """

    detections = dict_numpy2detections_list(pred)

    unique_labels = set([d.label for d in detections]).union([t.label for t in tracker_list])

    all_good_boxes = []
    all_good_trackers = []

    for id_ in unique_labels:

        trackers_list_id = [trk for trk in tracker_list if trk.label == id_]
        detections_id = [det for det in detections if det.label == id_]

        x_box = [trk.box for trk in trackers_list_id]
        z_box = [det.box for det in detections_id]


        matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(x_box, z_box, iou_thrd=0.3)

        # Deal with matched detections
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = trackers_list_id[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box = xx
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0

        # Deal with unmatched detections
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = Tracker()  # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            tmp_trk.id = id_incrementer
            tmp_trk.score = detections_id[idx].score
            tmp_trk.label = detections_id[idx].label
            id_incrementer+=1
            trackers_list_id.append(tmp_trk)
            x_box.append(xx)

        # Deal with unmatched tracks
        for trk_idx in unmatched_trks:
            tmp_trk = trackers_list_id[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            x_box[trk_idx] = xx


        good_boxes_id = []
        for trk in trackers_list_id:
            if ((trk.hits >= min_hits) and (trk.no_losses <= max_age)):
                good_boxes_id.append(Detection(trk.box, trk.label, trk.score, trk.id))
        all_good_boxes.extend(good_boxes_id)

        trackers_list_id = [x for x in trackers_list_id if x.no_losses <= max_age]
        all_good_trackers.extend(trackers_list_id)

    return detections_list2_dict_numpy(all_good_boxes), all_good_trackers, id_incrementer


def filter_pred_detections(pdp_iterable: Iterable[PredictionDatapoint]) -> Iterable[PredictionDatapoint]:
    """
    Assigns obj_id to predictions.

    Args:
        pdp_iterable: list or generator with PredictionDatapoint

    Yields:
        PredictionDatapoint with object ids
    """
    tracker_list = []
    id_incrementer = 0
    for pdp in pdp_iterable:
        filtered_pred, tracker_list, id_incrementer = pipeline(pdp.pred, tracker_list, id_incrementer=id_incrementer)
        yield PredictionDatapoint(filtered_pred, pdp.frame_id, pdp.reason)


def get_important_frames(df, labels_to_consider=None):
    """
    Given a dataframe having a set of tracked detections (obj_id is not None), return a partial dataframe containing the
    set of frames where all of unique objects are found when their size (area of the bounding box) in the image is at
    maximum.

    Args:
        df: pandas dataframe with obj_id column not None
        labels_to_consider: model class names ['car', 'truck' etc]

    Return:
        list of sorted important frame ids,
        dataframe with detections from the important frame ids
    """
    if labels_to_consider is None:
        labels_to_consider = ['truck', 'bus', 'train']
    df = df.dropna()
    df = df.astype({'obj_id': 'int32'})
    df['area'] = (df['x2'] - df['x1']) * (df['y2'] - df['y1'])

    bool_index = df['label'] == labels_to_consider[0]
    for c in labels_to_consider[1:]:
        bool_index = (bool_index) | (df['label'] == c)
    df = df[bool_index]
    df = df.set_index('img_id')

    important_frames = []
    for i in df['obj_id'].unique():
        important_frames.append(df[df['obj_id'] == i]['area'].idxmax())

    important_frames = sorted(list(set(important_frames)))

    df = df.loc[important_frames]
    # df['img_id'] = df.index
    df = df.reset_index()

    important_df = df

    return important_frames, important_df