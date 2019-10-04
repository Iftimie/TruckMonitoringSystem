from truckms.inference.neural import TruckDetector
import os.path as osp
import os
import cv2
from truckms.inference.utils import image_generator
from itertools import tee



def test_truck_detector():
    test_image = osp.join(osp.dirname(__file__), 'data', 'test_image.PNG')
    test_image = cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_BGR2RGB)
    p = TruckDetector()

    input_images = [test_image]
    predictions = list(p.compute(input_images))
    assert len(predictions) != 0
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], dict)
    assert 'labels' in predictions[0]
    assert 'scores' in predictions[0]
    assert 'boxes' in predictions[0]
    # cv2.imshow("image", next(p.plot_detections(input_images, predictions)))
    # cv2.waitKey(0)


# @pytest.mark.skip(reason="depends on local data")
def test_auu_data():
    auu_data_root = r'D:\aau-rainsnow\Hjorringvej\Hjorringvej-2'
    video_files = [osp.join(r, f) for (r, _, fs) in os.walk(auu_data_root) for f in fs if 'avi' in f or 'mkv' in f]
    p = TruckDetector()

    for video_path in video_files:
        image_gen = image_generator(video_path)
        image_gen1, image_gen2 = tee(image_gen)

        for image in p.plot_detections(image_gen1, p.compute(image_gen2)):
            cv2.imshow("image", image)
            cv2.waitKey(1)







