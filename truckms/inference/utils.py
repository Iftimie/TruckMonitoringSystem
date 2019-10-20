import cv2
from contextlib import contextmanager
from truckms.api import FrameDatapoint
from deprecated import deprecated


def framedatapoint_generator(video_path, skip=5, max_frames=-1) -> FrameDatapoint:
    """
    Generator function for processing images and keeping account of their frame ids

    Args:
        video_path: path to a video file. .avi, .mkv, .mp4 should work
        skip: number of frames to skip

    Yields:
        FrameDatapoint
    """
    assert skip >= 0
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    idx = 0
    while ret:
        if idx % (skip+1) == 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            yield FrameDatapoint(image, idx)
        idx += 1
        if idx == max_frames: break
        ret, image = cap.read()


@deprecated(reason="Not efficient. Use framedatapoint_generator_by_frame_ids2")
def framedatapoint_generator_by_frame_ids(video_path, frame_ids) -> FrameDatapoint:
    """
    Generator function for processing images. Only the frames whose ids appear in the frame_ids argument are yielded

    Args:
        video_path: path to a video file. .avi, .mkv, .mp4 should work
        frame_ids: list or set of frame ids to be returned. the iterable must be sorted

    Yields:
        FrameDatapoint
    """
    if len(frame_ids) == 0:
        return
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    idx = 0
    assert all(i < j for i, j in zip(frame_ids, frame_ids[1:]))
    while ret:
        if len(frame_ids) > 0 and idx == frame_ids[0]:
            frame_ids.pop(0)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            yield FrameDatapoint(image, idx)
        elif len(frame_ids) == 0:
            break

        idx += 1
        ret, image = cap.read()


def framedatapoint_generator_by_frame_ids2(video_path, frame_ids) -> FrameDatapoint:
    """
    Generator function for processing images. Only the frames whose ids appear in the frame_ids argument are yielded

    Args:
        video_path: path to a video file. .avi, .mkv, .mp4 should work
        frame_ids: list or set of frame ids to be returned. the iterable must be sorted

    Yields:
        FrameDatapoint
    """
    if len(frame_ids) == 0:
        return
    cap = cv2.VideoCapture(video_path)
    assert all(i < j for i, j in zip(frame_ids, frame_ids[1:]))
    for frame_num in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        res, image = cap.read()
        if res:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            yield FrameDatapoint(image, frame_num)


@contextmanager
def create_avi(path_to_avi, first_image):
    """
    Creates an avi file by receiving a fist image (for setting the output video resolution)

    Example usage:
    with create_avi(path_to_avi, image) as append_fn:
        for image in iterable:
            append_fn(image)

    Args:
        path_to_avi: path to avi
        first_image: numpy array
    """
    avi_file = cv2.VideoWriter(path_to_avi, cv2.VideoWriter_fourcc(*'DIVX'), 25, (first_image.shape[1], first_image.shape[0]))

    def append_fn(image):
        avi_file.write(image)

    try:
        yield append_fn
    finally:
        avi_file.release()
