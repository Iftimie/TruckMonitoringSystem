from flask import Blueprint, request, make_response, Flask
from functools import wraps, partial
from werkzeug import secure_filename
from flask import Blueprint, Flask, send_file, send_from_directory
from truckms.service.model import create_session, VideoStatuses, HeartBeats
import requests
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
    res = VideoStatuses.get_video_statuses(session).filter(VideoStatuses.results_path == None).all()

    if len(res) > 0:
        res.sort(key=lambda item: item.time_of_request)
        item = res[0]
        path = item.results_path
        session.close()
        if len(path.split(os.sep)) == 1:
            return send_from_directory(up_dir, path)
        else:
            return send_file(item.results_path)
    else:
        session.close()
        return make_response("Sorry, got no work to do", 404)

def upload_results(up_dir, db_url):
    for filename in request.files:
        f = request.files[filename]
        filename = secure_filename(filename)

        

        filepath = os.path.join(up_dir, filename)
        f.save(filepath)

def create_broker_blueprint(up_dir, db_url):
    broker_bp = Blueprint("broker_bp", __name__)
    heartbeat_func = (wraps(heartbeat)(partial(heartbeat, db_url)))
    broker_bp.route("/heartbeat", methods=['POST'])(heartbeat_func)
    broker_bp.role = "broker"
    return broker_bp


def create_broker_microservice(up_dir, db_url):
    app = Flask(__name__)
    app.roles = []
    broker_bp = create_broker_blueprint(up_dir, db_url)
    app.register_blueprint(broker_bp)
    app.roles.append(broker_bp.role)
    return app
