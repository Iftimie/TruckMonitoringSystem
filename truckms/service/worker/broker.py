from flask import Blueprint, request, make_response, Flask
from functools import wraps, partial
from werkzeug import secure_filename
from flask import Blueprint, Flask, send_file, send_from_directory
from truckms.service.model import create_session, VideoStatuses, HeartBeats
from truckms.service.worker.server import create_worker_microservice
import os


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


def create_broker_microservice(up_dir, db_url):
    # num_workers is 0 because this service is only a broker, however, a worker can also be a broker
    app, worker_pool = create_worker_microservice(up_dir, db_url, num_workers=1)
    worker_pool._processes = 0
    broker_bp = create_broker_blueprint(up_dir, db_url)
    app.register_blueprint(broker_bp)
    app.roles.append(broker_bp.role)
    return app, worker_pool
