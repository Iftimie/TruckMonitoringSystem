from truckms.inference.neural import create_model, pred_iter_to_pandas, compute
from truckms.inference.neural import create_model_efficient
from truckms.inference.utils import framedatapoint_generator
from truckms.inference.analytics import filter_pred_detections
from functools import partial
import os
from truckms.service.model import create_session, VideoStatuses
from flask import Blueprint, Flask, send_file, send_from_directory
from flask import request, make_response
from werkzeug import secure_filename
from functools import partial, wraps
import multiprocessing


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
    session.close()
    return destination


def upload_recordings(up_dir, db_url, worker_pool, analysis_func=None):
    """
    request must contain the file data and the options for running the detector
    max_operating_res, skip
    """
    for filename in request.files:
        f = request.files[filename]
        filename = secure_filename(filename)
        filepath = os.path.join(up_dir, filename)
        f.save(filepath)

        detector_options = request.form
        max_operating_res = detector_options['max_operating_res']
        skip = detector_options['skip']

        if analysis_func is None:
            analysis_func = partial(analyze_movie, max_operating_res=max_operating_res, skip=skip)

        res = worker_pool.apply_async(func=analyze_and_updatedb, args=(db_url, filepath, analysis_func))
        pass
    return make_response("Files uploaded and started runniing the detector. Check later for the results", 200)


def download_results(up_dir, db_url):
    """
    """
    session = create_session(db_url)
    filepath = os.path.join(up_dir, request.form["filename"])

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


def create_worker_blueprint(up_dir, db_url, num_workers, analysis_func=None):
    worker_pool = multiprocessing.Pool(num_workers)
    worker_bp = Blueprint("worker_bp", __name__)
    up_dir_func = (wraps(upload_recordings)(partial(upload_recordings, up_dir, db_url, worker_pool, analysis_func)))
    worker_bp.route("/upload_recordings", methods=['POST'])(up_dir_func)
    down_res_func = (wraps(download_results)(partial(download_results, up_dir, db_url)))
    worker_bp.route("/download_results", methods=['GET'])(down_res_func)
    worker_bp.role = "worker"
    return worker_bp, worker_pool


def create_worker_microservice(up_dir, db_url, num_workers):
    app = Flask(__name__)
    app.roles = []
    worker_bp, worker_pool = create_worker_blueprint(up_dir, db_url, num_workers)
    app.register_blueprint(worker_bp)
    app.roles.append(worker_bp.role)
    return app, worker_pool
