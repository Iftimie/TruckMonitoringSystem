import cv2
from contextlib import contextmanager

def image_generator(video_path, skip=5):
    """
    skip 0 is valid
    """
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    idx = 0
    while ret:
        if idx % (skip+1) ==0:
            yield image, idx
        idx+=1
        ret, image = cap.read()


def image_generator_by_frame_ids(video_path, frame_ids):
    """
    skip 0 is valid
    """
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    idx = 0
    while ret:
        if idx in frame_ids:
            yield image, idx
        idx+=1
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
