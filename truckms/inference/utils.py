import cv2


def image_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    c = 0
    while ret:
        if c % 5 ==0:
            yield image
        c+=1
        ret, image = cap.read()