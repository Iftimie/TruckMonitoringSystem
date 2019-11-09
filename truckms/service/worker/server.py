from truckms.inference.neural import create_model, pred_iter_to_pandas, compute
from truckms.inference.neural import create_model_efficient
from truckms.inference.utils import framedatapoint_generator
from truckms.inference.analytics import filter_pred_detections
from functools import partial
import os
from truckms.service.model import create_session, VideoStatuses


def analyze_movie(video_path, max_operating_res, skip=0):
    """
    Attention!!! if the movie is short or too fast and skip  is too big, then it may result with no detections
    #TODO think about this
    """
    model = create_model_efficient(model_creation_func=partial(create_model, max_operating_res=max_operating_res))
    image_gen = framedatapoint_generator(video_path, skip=skip)
    pred_gen = compute(image_gen, model=model, batch_size=5)
    filtered_pred = filter_pred_detections(pred_gen)
    df = pred_iter_to_pandas(filtered_pred)
    destination = os.path.splitext(video_path)[0]+'.csv'
    df.to_csv(destination)
    return destination


def analyze_and_updatedb(db_url, video_path, analysis_func):
    """
    Args:
        db_url: url for database
        video_path: path to a file on the local disk
        analysis_func: a function that receives an argument with the video path and returns the path to results.csv
    """
    session = create_session(db_url)
    VideoStatuses.add_video_status(session, file_path=video_path, results_path=None)
    destination = analysis_func(video_path)
    VideoStatuses.update_results_path(session, file_path=video_path, new_results_path=destination)




def create_worker_blueprint():
    pass