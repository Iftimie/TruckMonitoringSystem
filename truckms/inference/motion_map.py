import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import default_collate
import torch
from itertools import tee
from truckms.inference.utils import framedatapoint_generator
import cv2
import numpy as np


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


def pytorch_motion_map(video_file):
    """
    Not very useful. it's slow.
    """
    smoothing = GaussianSmoothing(1, 5, 1)
    batch_size = 50

    image_gen = framedatapoint_generator(video_file, skip=0, grayscale=True)
    image_gen1, image_gen2 = tee(image_gen)

    diff_skip = 7
    for i in range(diff_skip):
        next(image_gen2)  # skip the second one

    # https://stackoverflow.com/questions/56235733/is-there-a-tensor-operation-or-function-in-pytorch-that-works-like-cv2-dilate
    # https://docs.opencv.org/2.4/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html
    with torch.no_grad():
        while True:
            try:
                list1 = [next(image_gen1).image for _ in range(batch_size)]
                list2 = [next(image_gen2).image for _ in range(batch_size)]
                tensor1 = default_collate(list1).type(torch.int32)
                tensor2 = default_collate(list2).type(torch.int32)
                # diff = (tensor2 - tensor1).abs()
                # diff = diff.view(diff.shape[0], -1).sum(dim=1)

                tensor1 = tensor1.view(tensor1.shape[0], 1, tensor1.shape[1], tensor1.shape[2])
                tensor2 = tensor2.view(tensor2.shape[0], 1, tensor2.shape[1], tensor2.shape[2])
                tensor1 = smoothing(tensor1.float())
                tensor2 = smoothing(tensor2.float())

                diff = (tensor1 - tensor2).abs()
                diff = (diff > 50).type(torch.float32)
                diff = (-nn.functional.max_pool2d(-diff, 3, 1))  # erosion
                diff = (-nn.functional.max_pool2d(-diff, 3, 1))  # erosion

                # diff = np.squeeze((diff*255).numpy().astype(np.uint8))
                # for original, image in zip(diff, list1):
                #     cv2.imshow("image", image)
                #     cv2.imshow("original", original)
                #     cv2.waitKey(1)
            except StopIteration:
                break
        pass


def movement_generator(video_test, diff_skip=7, erosion=2, binary_threshold=50):
    """
    Generator that yields movement values. The number yielded of frame_ids will be the same as the video length.
    Movement values are computed basically as the difference between 2 frames separated by diff_skip other frames.
    Some cleaning morphological operations are applied for bettwe results.

    Args:
        video_test: path to video file
        diff_skip: difference in time (ids) between the frames that have to be subtracted
        erosion: erosion iterations. higher value will result in smaller movement values as more pixels are erased
        binary_threshold: the difference between pixels of two frames will be accepted for futher processing if the pixels
            is higher than a threshold

    Yields:
        integer movement value
    """
    image_gen = framedatapoint_generator(video_test, skip=0, grayscale=True)
    image_gen1, image_gen2 = tee(image_gen)

    for i in range(diff_skip):
        next(image_gen2)  # skip the second one

    for fdp1, fdp2 in zip(image_gen1, image_gen2):

        diff = cv2.absdiff(fdp1.image, fdp2.image)
        diff = cv2.threshold(diff, binary_threshold, 255, cv2.THRESH_BINARY)[1]
        diff = cv2.erode(diff, None, iterations=erosion)

        movement = diff.sum() // 255
        yield movement

    for i in range(diff_skip):
        yield 0


def motion_filter(movement_gen, threshold=1000):
    """
    Generator filtering the movement values. Higher value will result in fewer accpted frames.
    """
    for movement in movement_gen:
        if movement > threshold:
            yield 1
        else:
            yield 0


def motion_expander(array_motion, kernel_size=35):
    """
    Once a motion is detected at frame i we may want to expand the search and also take into consideration the next
        'kernel_size' frames.
    """
    # res = (np.convolve(array_motion, [1] * kernel_size, 'same') > 0).astype(np.uint8)
    new_array = [0] * array_motion.shape[0]
    for idx, val in enumerate(array_motion):
        if val == 1:
            for i in range(kernel_size + 1):
                new_array[idx+i] = 1
    return np.array(new_array)


def movement_frames_indexes(video_test, movement_gen=movement_generator, motion_filter=motion_filter):
    """
    Function that finds frames in a video file where motion is present in the sequence.
    Args:
        video_test: file to be analyzed
        movement_gen: a generator that receives the video file and yields values representing the movement value in a frame
        motion_filter: a generator coupled to the previous that filters out movement values that are under a threshold

    Returns:
        array of frame ids where motion is present
    """
    movement_array = np.array([motion_presence for motion_presence in motion_filter(movement_gen(video_test))])
    expanded_array = motion_expander(movement_array)
    res = np.where(expanded_array)[0]
    return res

