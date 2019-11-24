from truckms.inference.neural import create_model, pred_iter_to_pandas, compute
from truckms.inference.neural import create_model_efficient
from truckms.inference.utils import framedatapoint_generator
from truckms.inference.analytics import filter_pred_detections
import os
from truckms.service.model import create_session, VideoStatuses, HeartBeats
from flask import Blueprint, Flask, send_file, send_from_directory
from flask import request, make_response
from werkzeug import secure_filename
from functools import partial, wraps
import multiprocessing
import logging
import traceback
from truckms.service_v2.api import P2PFlaskApp, P2PBlueprint
from typing import Callable
logger = logging.getLogger(__name__)


def analyze_movie(video_path, max_operating_res, skip=0):
    """
    Attention!!! if the movie is short or too fast and skip  is too big, then it may result with no detections
    #TODO think about this

    Args:
        video_path: path to a file on the local disk
        max_operating_res: the resolution used for analyzing the video file. The lower the resolution, the earlier it
            will finish but with less accuracy
        skip: number of frames to skip. the higher the number, the earlier the function will finish

    Return:
          path to results file
    """
    model = create_model_efficient(model_creation_func=partial(create_model, max_operating_res=max_operating_res))
    image_gen = framedatapoint_generator(video_path, skip=skip)
    pred_gen = compute(image_gen, model=model, batch_size=5)
    filtered_pred = filter_pred_detections(pred_gen)
    df = pred_iter_to_pandas(filtered_pred)
    destination = os.path.splitext(video_path)[0]+'.csv'
    df.to_csv(destination)
    return destination


def analyze_and_updatedb(db_url: str, video_path: str, analysis_func: Callable[[str], str]):
    """
    Args:
        db_url: url for database
        video_path: path to a file on the local disk
        analysis_func: a function that receives an argument with the video path and returns the path to results.csv

    Return:
        path to results file
    """
    destination = None
    try:
        session = create_session(db_url)
        VideoStatuses.add_video_status(session, file_path=video_path, results_path=None)
        destination = analysis_func(video_path)
        VideoStatuses.update_results_path(session, file_path=video_path, new_results_path=destination)
        session.close()
    except:
        logger.info(traceback.format_exc())
    return destination


def pool_can_do_more_work(worker_pool):
    """
    Checks if there are available workers in the pool.
    """
    count_opened = 0
    for apply_result in worker_pool.futures_list[:]:
        try:
            apply_result.get(1)
            # if not timeout exception, then we can safely remove the object
            worker_pool.futures_list.remove(apply_result)
        except:
            count_opened+=1
    if count_opened < worker_pool._processes:
        return True
    else:
        return False


def worker_heartbeats(db_url):
    """
    If a client worker (worker that cannot be reachable and asks a broker for work) is available, then it will send a
    signal to the broker and ask for work.
    """
    session = create_session(db_url)
    boolean = HeartBeats.has_recent_heartbeat(session, minutes=20)
    session.close()
    return boolean


def upload_recordings(up_dir, db_url, worker_pool, analysis_func=None):
    """
    request must contain the file data and the options for running the detector
    max_operating_res, skip

    #TODO refactor the stuff with worker_heartbeats into a analysis func that is added by default by the broker.
    """
    for filename in request.files:
        f = request.files[filename]
        filename = secure_filename(filename)
        filepath = os.path.join(up_dir, filename)
        f.save(filepath)

        detector_options = request.form
        max_operating_res = int(detector_options['max_operating_res'])
        skip = int(detector_options['skip'])

        #TODO move this part into analysis func in broker
        if not pool_can_do_more_work(worker_pool) and worker_heartbeats(db_url):
            #store the files as broker
            session = create_session(db_url)
            VideoStatuses.add_video_status(session, file_path=filepath, max_operating_res=max_operating_res, skip=skip)
            # this ? time of request will be updated both at uploading and at dispatching to worker. we want to serve the oldest request that does not have a results path
            # and we need to know which one is the most ignored
            # or
            # this ?time of request will be set only when a worker asks for this file
            session.close()
        #TODO move this part into analysis func in broker
        else:
            if analysis_func is None:
                analysis_func = partial(analyze_movie, max_operating_res=max_operating_res, skip=skip)

            res = worker_pool.apply_async(func=analyze_and_updatedb, args=(db_url, filepath, analysis_func))
            worker_pool.futures_list.append(res)
    return make_response("Files uploaded and started runniing the detector. Check later for the results", 200)


def download_results(up_dir, db_url):
    """
    Downloads the results of analysis

    Args:
        up_dir: path to directory where to store the uploaded video files
        db_url: url to database
    """
    session = create_session(db_url)
    filepath = os.path.join(up_dir, request.form["filename"])
    session.close()
    try:
        item = VideoStatuses.find_results_path(session, filepath)
        if item.results_path is not None:
            if len(item.results_path.split(os.sep)) == 1:
                return send_from_directory(up_dir, item.results_path)
            else:
                return send_file(item.results_path)
        else:
            return make_response("File still processing", 202)
    except:
        return make_response("There is no file with this name: "+request.form["filename"], 404)


class P2PWorkerBlueprint(P2PBlueprint):

    def __init__(self, *args, num_workers, **kwargs):
        super(P2PWorkerBlueprint, self).__init__(*args, **kwargs)
        self.worker_pool = multiprocessing.Pool(num_workers)
        self.worker_pool.futures_list = []


def create_worker_p2pblueprint(up_dir: str, db_url: str, num_workers: int,
                               analysis_func: Callable[[str], str] = None) -> P2PBlueprint:
    """
    Creates a P2PBlueprint. The worker blueprint has the responsibility of responding to requests of uploading video files
    in order to be analyzed and requests of downloading the analysis results.

    Args:
        up_dir: path to directory where to store the uploaded video files
        db_url: url to database
        num_workers: number of workers for processing video files in parallel
        analysis_func: callable that receives the path to a video file. the function should return the path to the results file
    """

    worker_bp = P2PWorkerBlueprint("worker_bp", __name__, num_workers=num_workers, role="worker")
    up_dir_func = (wraps(upload_recordings)(partial(upload_recordings, up_dir, db_url, worker_bp.worker_pool, analysis_func)))
    worker_bp.route("/upload_recordings", methods=['POST'])(up_dir_func)
    down_res_func = (wraps(download_results)(partial(download_results, up_dir, db_url)))
    worker_bp.route("/download_results", methods=['GET'])(down_res_func)

    return worker_bp


def create_worker_service(up_dir, db_url, num_workers):
    # TODO should I create different databases for workers, brokers, etc???
    #  in order to avoid conflict between workers and brokers?
    app = P2PFlaskApp(__name__)
    worker_bp = create_worker_p2pblueprint(up_dir, db_url, num_workers)
    app.register_blueprint(worker_bp)
    return app
