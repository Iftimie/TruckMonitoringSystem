import cv2


def image_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    while ret:
        yield image
        ret, image = cap.read()