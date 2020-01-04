import os.path as osp
from truckms.inference.utils import framedatapoint_generator
from itertools import tee
import cv2
import numpy as np
from torch.utils.data.dataloader import default_collate
import torch
import time


def faster_pytorch(g1, g2):
    batch_size = 120

    # https://stackoverflow.com/questions/56235733/is-there-a-tensor-operation-or-function-in-pytorch-that-works-like-cv2-dilate
    # https://docs.opencv.org/2.4/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html

    while True:
        try:
            list1 = [next(g1).image for _ in range(batch_size)]
            list2 = [next(g2).image for _ in range(batch_size)]
            tensor1 = default_collate(list1).type(torch.int32)
            tensor2 = default_collate(list2).type(torch.int32)
            diff = (tensor2 - tensor1).abs()
            diff = diff.view(diff.shape[0], -1).sum(dim=1)

            diff = diff.numpy()
            # for image in diff:
            #     cv2.imshow("image", image)
            #     cv2.waitKey(1)
        except StopIteration:
            break
    pass


def test_motion_map_slow():

    video_test = osp.join(osp.dirname(__file__), "..", "service", "data", "cut.mkv")
    video_test = r'D:\tms_data\good_data\good_datadontdelete2\output_18.mp4'

    image_gen = framedatapoint_generator(video_test, skip=0, max_frames=2400, grayscale=True)
    image_gen1, image_gen2 = tee(image_gen)
    next(image_gen2)  # skip the second one

    # some thresholds should be put. for maximum, average??, erosion iterations or kernel, and max_length after motion detection

    start = time.time()

    for fdp1, fdp2 in zip(image_gen1, image_gen2):
        # cv2.imshow("fdp1", fdp1.image)
        # cv2.imshow("fdp2", fdp2.image)

        fdp1.image = cv2.GaussianBlur(fdp1.image, (3, 3), 0)
        fdp2.image = cv2.GaussianBlur(fdp2.image, (3, 3), 0)

        diff = cv2.absdiff(fdp1.image, fdp2.image)
        # diff = np.abs(fdp1.image.astype(np.int32) - fdp2.image.astype(np.int32))
        # diff = (diff > 50) * 255

        diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]

        diff = cv2.erode(diff, None, iterations=2)
        # diff = cv2.dilate(diff, (3,3), iterations=1)

        diff_copy = diff.copy()
        sum_ = diff.sum() // 255
        avg_ = np.average(diff)
        diff_copy = cv2.putText(diff_copy, "{}, {}".format(sum_, avg_), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255))

        # flow = cv2.calcOpticalFlowFarneback(fdp1.image, fdp2.image, None, 0.5, 3, 100, 3, 5, 1.2, 0)
        # flow =flow[:,:,0]

        cv2.imshow("diff", diff_copy)
        cv2.imshow("fdp1.image", fdp1.image)
        # cv2.imshow("flow", flow)
        cv2.waitKey(100)

    end = time.time()
    print(end-start)


def test_motion_map_faster():
    video_test = osp.join(osp.dirname(__file__), "..", "service", "data", "cut.mkv")
    video_test = r'D:\tms_data\good_data\good_datadontdelete2\output_18.mp4'

    start = time.time()
    image_gen = framedatapoint_generator(video_test, skip=0, max_frames=2400, grayscale=True)
    image_gen1, image_gen2 = tee(image_gen)
    next(image_gen2)  # skip the second one

    start = time.time()
    faster_pytorch(image_gen1, image_gen2)
    end = time.time()
    print(end-start)