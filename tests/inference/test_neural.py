from truckms.inference.neural import create_model, compute, plot_detections, pred_iter_to_pandas, pandas_to_pred_iter
from truckms.inference.neural import iterable2batch
from truckms.inference.utils import framedatapoint_generator_by_frame_ids2
from truckms.api import FrameDatapoint, PredictionDatapoint
import os.path as osp
import os
import cv2
from truckms.inference.utils import framedatapoint_generator
from itertools import tee


def test_iterable2batch():
    g1 = framedatapoint_generator_by_frame_ids2(video_path=osp.join(osp.dirname(__file__),
                                                                    '..', 'service', 'data', 'cut.mkv'),
                                      frame_ids=[3, 6, 10, 12, 17])
    g2 = iterable2batch(g1, batch_size=2)

    for idx, bfdp in enumerate(g2):
        if idx != 2:
            assert len(bfdp.batch_images) == 2
        else:
            assert len(bfdp.batch_images) == 1


def test_truck_detector():
    test_image = osp.join(osp.dirname(__file__), 'data', 'test_image.PNG')
    test_image = cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_BGR2RGB)
    model = create_model()

    input_images = [FrameDatapoint(test_image, 1)]
    predictions = list(compute(input_images, model))
    assert len(predictions) != 0
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], PredictionDatapoint)
    assert isinstance(predictions[0].pred, dict)
    assert 'labels' in predictions[0].pred
    assert 'scores' in predictions[0].pred
    assert 'boxes' in predictions[0].pred
    # cv2.imshow("image", next(p.plot_detections(input_images, predictions)))
    # cv2.waitKey(0)


# @pytest.mark.skip(reason="depends on local data")
def test_auu_data():
    auu_data_root = r'D:\aau-rainsnow\Hjorringvej\Hjorringvej-2'
    video_files = [osp.join(r, f) for (r, _, fs) in os.walk(auu_data_root) for f in fs if 'avi' in f or 'mkv' in f]
    model = create_model()

    for video_path in video_files:
        image_gen = framedatapoint_generator(video_path, skip=5)
        image_gen1, image_gen2 = tee(image_gen)

        for idx, fdp in enumerate(plot_detections(image_gen1, compute(image_gen2, model))):
            cv2.imshow("image", fdp.image)
            cv2.waitKey(1)
            if idx==5:break

def test_TruckDetector_pred_iter_to_pandas():
    auu_data_root = r'D:\aau-rainsnow\Hjorringvej\Hjorringvej-2'
    video_file = [osp.join(r, f) for (r, _, fs) in os.walk(auu_data_root) for f in fs if 'avi' in f or 'mkv' in f][0]
    #file 'Hjorringvej\\Hjorringvej-2\\cam1.mkv' has 6000 frames
    model = create_model(max_operating_res=320)
    image_gen = framedatapoint_generator(video_file, skip=6000//30)
    image_gen1, image_gen2 = tee(image_gen)

    pred_gen = compute(image_gen1, model)

    df = pred_iter_to_pandas(pred_gen)

    pred_gen_from_df = pandas_to_pred_iter(df)

    for idx, fdp in enumerate(plot_detections(image_gen2, pred_gen_from_df)):
        cv2.imshow("image", fdp.image)
        cv2.waitKey(1)
        if idx==5:break

