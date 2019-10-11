import cv2
from contextlib import contextmanager


def image_id_yielder(f):
    """
    Decorator for clarity purposes. It serves as declaring that a generator or a function has a specific interface
    You can define a generator that yields an image and id and annotate with this decorator just for clarity purposes.
    #TODO It should  also check that the first yield does indeed yields an array and an integer

    Example:
        @image_id_yielder
        def generator():
            yield np.random((100,100)), 0

    Args:
        f: generator function to be decorated

    Return:
         decorated function
    """
    return f


def batch_image_id_yielder(f):
    #TODO same as image_id_yielder but it returns a batch of elements
    return f


def prediction_id_yielder(f):
    return f


@image_id_yielder
def image_generator(video_path, skip=5):
    """
    Generator function for processing images and keeping account of their frame ids

    Args:
        video_path: path to a video file. .avi, .mkv, .mp4 should work
        skip: number of frames to skip

    Yields:
        image and id
    """
    assert skip >= 0
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    idx = 0
    while ret:
        if idx % (skip+1) == 0:
            yield image, idx
        idx += 1
        ret, image = cap.read()


@image_id_yielder
def image_generator_by_frame_ids(video_path, frame_ids):
    """
    Generator function for processing images. Only the frames whose ids appear in the frame_ids argument are yielded

    Args:
        video_path: path to a video file. .avi, .mkv, .mp4 should work
        frame_ids: list or set of frame ids to be returned

    Yields:
        image and frame id
    """
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    idx = 0
    frame_ids = set(frame_ids)
    while ret:
        if idx in frame_ids:
            yield image, idx
        idx += 1
        ret, image = cap.read()


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
