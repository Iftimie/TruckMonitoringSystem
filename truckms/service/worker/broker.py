from flask import Blueprint, request, make_response, Flask
from functools import wraps, partial
from werkzeug import secure_filename
from flask import Blueprint, Flask, send_file, send_from_directory
from truckms.service.model import create_session, VideoStatuses, HeartBeats
import os
from truckms.service_v2.api import P2PFlaskApp
from truckms.service.worker.server import create_worker_p2pblueprint
from truckms.service.worker import server
from truckms.service.worker.server import analyze_movie, analyze_and_updatedb


def heartbeat(db_url):
    """
    Pottential vulnerability from flooding here
    """
    session = create_session(db_url)
    HeartBeats.add_heartbeat(session)
    session.close()
    return make_response("Thank god you are alive", 200)


def download_recordings(up_dir, db_url):
    session = create_session(db_url)
    # TODO there are some edge cases that I havent treated. an allready downloaded recording might not get a results because of a dead worker
    #  on the line below that case won't get selected for reprocessing because it has been asigned a remote_ip
    #  or maybe I should not assign remote_ip and remote_port and allow for race conditions?
    # res = VideoStatuses.get_video_statuses(session).filter(VideoStatuses.results_path == None,
    #                                                        VideoStatuses.remote_ip != None,
    #                                                        VideoStatuses.remote_port != None).all()
    res = session.query(VideoStatuses).filter(VideoStatuses.results_path == None).all()
    heartbeat(db_url)
    session.close()
    # TODO may remove the route for heartbeat as it is redundant
    #  if a worker asks for a file, It should automatically add a heat bead.
    if len(res) > 0:
        res.sort(key=lambda item: item.time_of_request)
        item = res[0]
        path = item.file_path

        if len(path.split(os.sep)) == 1:
            result = send_from_directory(up_dir, path, as_attachment=True)
        else:
            result = send_file(path, as_attachment=True)

        result.headers["max_operating_res"] = item.max_operating_res
        result.headers["skip"] = item.skip
        result.headers["filename"] = os.path.basename(item.file_path)
        return result

    else:
        return make_response("Sorry, got no work to do", 404)


def upload_results(up_dir, db_url):
    for filename in request.files:
        f = request.files[filename]
        filename = secure_filename(filename)
        filepath = os.path.join(up_dir, filename)
        f.save(filepath)
        session = create_session(db_url)
        VideoStatuses.update_results_path(session, file_path=None, new_results_path=filepath)
        session.close()
    return make_response("Thanks for your precious work", 200)


def create_broker_blueprint(up_dir, db_url):
    broker_bp = Blueprint("broker_bp", __name__)
    heartbeat_func = (wraps(heartbeat)(partial(heartbeat, db_url)))
    broker_bp.route("/heartbeat", methods=['POST'])(heartbeat_func)

    up_res_func = (wraps(upload_results)(partial(upload_results, up_dir, db_url)))
    broker_bp.route("/upload_results", methods=['POST'])(up_res_func)

    down_rec_func = (wraps(download_recordings)(partial(download_recordings, up_dir, db_url)))
    broker_bp.route("/download_recordings", methods=['GET'])(down_rec_func)

    broker_bp.role = "broker"
    return broker_bp


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
    Overwritten route from truckms.service.worker.server
    """
    for filename in request.files:
        f = request.files[filename]
        filename = secure_filename(filename)
        filepath = os.path.join(up_dir, filename)
        f.save(filepath)

        detector_options = request.form
        max_operating_res = int(detector_options['max_operating_res'])
        skip = int(detector_options['skip'])

        if not pool_can_do_more_work(worker_pool) and worker_heartbeats(db_url):
            # store the files as broker
            session = create_session(db_url)
            VideoStatuses.add_video_status(session, file_path=filepath, max_operating_res=max_operating_res, skip=skip)
            # this ? time of request will be updated both at uploading and at dispatching to worker. we want to serve the oldest request that does not have a results path
            # and we need to know which one is the most ignored
            # or
            # this ?time of request will be set only when a worker asks for this file
            session.close()
        else:
            if analysis_func is None:
                analysis_func = partial(analyze_movie, max_operating_res=max_operating_res, skip=skip)
        if analysis_func is None:
            analysis_func = partial(analyze_movie, max_operating_res=max_operating_res, skip=skip)

            res = worker_pool.apply_async(func=analyze_and_updatedb, args=(db_url, filepath, analysis_func))
            worker_pool.futures_list.append(res)
        res = worker_pool.apply_async(func=analyze_and_updatedb, args=(db_url, filepath, analysis_func))
        worker_pool.futures_list.append(res)
    return make_response("Files uploaded and started runniing the detector. Check later for the results", 200)


def create_broker_microservice(up_dir, db_url, num_workers=0) -> P2PFlaskApp:
    """
    Args:
        up_dir: path to directory where to store video files and files with results
        db_url: url to database
        num_workers: number of workers for the worker_blueprint. In this case is 0 because this service is by default
        only a broker, however, a worker_blueprint can also be a broker and not have num_workers set on 0

    Return:
        P2PFlaskApp
    """
    app = P2PFlaskApp(__name__)
    worker_bp = create_worker_p2pblueprint(up_dir, db_url, num_workers=1 if num_workers == 0 else num_workers)

    # Overwriting the /upload_recordings rule from create_worker_p2pblueprint
    up_dir_func = (
        wraps(upload_recordings)(partial(upload_recordings, up_dir, db_url, worker_bp.worker_pool)))
    worker_bp.route("/upload_recordings", methods=['POST'])(up_dir_func)
    if num_workers == 0:
        worker_bp.worker_pool._processes = 0
    app.register_blueprint(worker_bp)

    broker_bp = create_broker_blueprint(up_dir, db_url)
    app.register_blueprint(broker_bp)
    return app
