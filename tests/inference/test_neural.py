from truckms.inference.neural import create_model, compute, plot_detections, pred_iter_to_pandas, pandas_to_pred_iter
import os.path as osp
import os
import cv2
from truckms.inference.utils import image_generator
from itertools import tee


def test_truck_detector():
    test_image = osp.join(osp.dirname(__file__), 'data', 'test_image.PNG')
    test_image = cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_BGR2RGB)
    model = create_model()

    input_images = [(test_image, 1)]
    predictions = list(compute(input_images, model))
    assert len(predictions) != 0
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], tuple)
    assert isinstance(predictions[0][0], dict)
    assert 'labels' in predictions[0][0]
    assert 'scores' in predictions[0][0]
    assert 'boxes' in predictions[0][0]
    # cv2.imshow("image", next(p.plot_detections(input_images, predictions)))
    # cv2.waitKey(0)


# @pytest.mark.skip(reason="depends on local data")
def test_auu_data():
    auu_data_root = r'D:\aau-rainsnow\Hjorringvej\Hjorringvej-2'
    video_files = [osp.join(r, f) for (r, _, fs) in os.walk(auu_data_root) for f in fs if 'avi' in f or 'mkv' in f]
    model = create_model()

    for video_path in video_files:
        image_gen = image_generator(video_path, skip=5)
        image_gen1, image_gen2 = tee(image_gen)

        for idx, (image, _) in enumerate(plot_detections(image_gen1, compute(image_gen2, model))):
            cv2.imshow("image", image)
            cv2.waitKey(1)
            if idx==5:break

def test_TruckDetector_pred_iter_to_pandas():
    auu_data_root = r'D:\aau-rainsnow\Hjorringvej\Hjorringvej-2'
    video_file = [osp.join(r, f) for (r, _, fs) in os.walk(auu_data_root) for f in fs if 'avi' in f or 'mkv' in f][0]
    #file 'Hjorringvej\\Hjorringvej-2\\cam1.mkv' has 6000 frames
    model = create_model(max_operating_res=320)
    image_gen = image_generator(video_file, skip=6000//30)
    image_gen1, image_gen2 = tee(image_gen)

    pred_gen = compute(image_gen1, model)

    df = pred_iter_to_pandas(pred_gen)

    pred_gen_from_df = pandas_to_pred_iter(df)

    for idx, (image, _) in enumerate(plot_detections(image_gen2, pred_gen_from_df)):
        cv2.imshow("image", image)
        cv2.waitKey(1)
        if idx==5:break







