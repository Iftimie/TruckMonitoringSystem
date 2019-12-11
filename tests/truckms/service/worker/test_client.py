from truckms.service.bookkeeper import ServerThread
from truckms.service.bookkeeper import create_bookkeeper_service, create_bookkeeper_p2pblueprint
from truckms.service.worker.server import create_worker_p2pblueprint
from truckms.service import bookkeeper
import requests
from mock import Mock
from flask import Flask
import os


def dummy_analysis_func(video_path):
    _, file_extension = os.path.splitext(video_path)
    csv_path = video_path.replace(file_extension, ".csv")
    with open(csv_path, "w") as f:
        f.write("dummy_content")
    return csv_path


def create_bookkeeper_worker_ms(tmpdir, number):
    db_url = 'sqlite:///' + os.path.join(tmpdir.strpath, "database"+ str(number)+".sqlite")
    up_dir = os.path.join(tmpdir.strpath, "updir" + str(number))
    os.mkdir(up_dir)
    app = Flask(__name__)
    app.roles = []
    bookkeeper_bp = create_bookkeeper_p2pblueprint()
    app.register_blueprint(bookkeeper_bp)
    # app.roles.append(bookkeeper_bp.role)

    worker_bp, _ = create_worker_p2pblueprint(up_dir, db_url, 1, analysis_func=dummy_analysis_func)
    app.register_blueprint(worker_bp)
    app.roles.append(bookkeeper_bp.role)
    #TODO refactor this into classes or something functional.
    # no need to do register and roles.append all the time
    return app


def create_bookkeeper_worker_servers(tmpdir):
    r"""Look in tests\service\bookkeeper\test_servic.py for details about the communication"""
    bookkeeper.find_workload = Mock(return_value=0)
    server1 = ServerThread(create_bookkeeper_worker_ms(tmpdir, 0), port=5000)
    server1.start()  # at ths step there is only one node state, and only server1 knows about it (about itself)

    bookkeeper.find_workload = Mock(return_value=100)
    server2 = ServerThread(create_bookkeeper_worker_ms(tmpdir, 1), port=5001, central_host='127.0.0.1', central_port=5000)
    server2.start()  # at this step server 1 knows about itself and server 2
    # server 2 knows about itself and server 1
    bookkeeper.find_workload = Mock(return_value=50)
    server3 = ServerThread(create_bookkeeper_worker_ms(tmpdir, 2), port=5002, central_host='127.0.0.1', central_port=5000)
    server3.start()  # at this step server 1 knows about itself, server 2 and server 3
    # server 2 knows about itself and server 1
    # server 3 knows about itself, server 1 and server 2
    # server 2 will make a network discovery
    res2 = requests.get('http://localhost:5001/node_states').json()  # will get the data defined above
    discovered_states = []
    for state in res2:
        discovered_states += requests.get('http://{}:{}/node_states'.format(state['ip'], state[
            'port'])).json()  # TODO should rename everything from host to ip
    server2.client.post("/node_states", json=discovered_states)

    return [server1, server2, server3]


def shutdown_servers(servers):
    for i in servers:
        i.shutdown()


