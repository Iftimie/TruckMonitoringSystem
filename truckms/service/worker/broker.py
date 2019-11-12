from flask import Blueprint, request, make_response
from functools import wraps, partial
from werkzeug import secure_filename
from truckms.service.model import create_session, VideoStatuses
import requests
import os


def upload_recordings(up_dir, db_url):
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

        session = create_session(db_url)
        VideoStatuses.add_video_status(session, file_path=filepath, max_operating_res=max_operating_res, skip=skip,
                                       time_of_request=None) # time of request will be set only when a worker asks for this file

    return make_response("Files uploaded and started runniing the detector. Check later for the results", 200)


def hearbeat_from_worker():
    # any heartbeat from a worker will update the workload of the current node, and set it to 0

    res1 = requests.get('http://localhost:5000/node_states').json()  # will get the data defined above
    res1 = [res1]

    # I should search here for 0.0.0.0 not 127.0.0.1
    # TODO 

    data = [{'ip': '127.0.0.1', 'port': self.port, 'workload': find_workload(), 'hardware': "Nvidia GTX 960M Intel i7",
             'nickname': "rmstn",
             'node_type': ",".join(app.roles), 'email': 'iftimie.alexandru.florentin@gmail.com'}]
    # register self state to local service
    res = self.client.post("/node_states", json=data)


def create_worker_blueprint(up_dir, db_url):

    worker_bp = Blueprint("broker_bp", __name__)
    up_dir_func = (wraps(upload_recordings)(partial(upload_recordings, up_dir, db_url)))
    worker_bp.route("/upload_recordings", methods=['POST'])(up_dir_func)


    down_res_func = (wraps(download_results)(partial(download_results, up_dir, db_url)))
    worker_bp.route("/download_results", methods=['GET'])(down_res_func)
    worker_bp.role = "worker"
    return worker_bp, worker_pool