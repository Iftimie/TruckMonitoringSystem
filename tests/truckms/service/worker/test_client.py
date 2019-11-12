from truckms.service.bookkeeper import ServerThread
from truckms.service.bookkeeper import create_microservice, create_bookkeeper_blueprint
from truckms.service.worker.server import create_worker_blueprint
from truckms.service import bookkeeper
import requests
from truckms.service.worker.user_client import select_lru_worker
from truckms.service.worker import user_client
from mock import Mock
import os.path as osp
from truckms.service.model import create_session, VideoStatuses
from truckms.service import worker
import time
from flask import Flask
import os

def test_select_lru_worker():
    bookkeeper.find_workload = Mock(return_value=100)
    server1 = ServerThread(create_microservice(), port=5000)
    server1.start()
    bookkeeper.find_workload = Mock(return_value=50)
    server2 = ServerThread(create_microservice(), port=5001, central_host='127.0.0.1', central_port=5000)
    server2.start()

    res1 = requests.get('http://localhost:5000/node_states').json()  # will get the data defined above
    res2 = requests.get('http://localhost:5001/node_states').json()  # will get the data defined above
    assert len(res1) == 2
    assert len(res2) == 2
    assert res1 == res2

    assert sorted(res1, key=lambda x: x['port']) == [
        {'ip': '127.0.0.1', 'port': 5000, 'workload': 100, 'hardware': "Nvidia GTX 960M Intel i7", 'nickname': "rmstn",
         'node_type': "bookkeeper", 'email': 'iftimie.alexandru.florentin@gmail.com'},
        {'ip': '127.0.0.1', 'port': 5001, 'workload': 50, 'hardware': "Nvidia GTX 960M Intel i7", 'nickname': "rmstn",
         'node_type': "bookkeeper", 'email': 'iftimie.alexandru.florentin@gmail.com'}]

    lru_ip, lru_port = select_lru_worker()
    assert lru_ip == '127.0.0.1'
    assert lru_port == 5001

    server1.shutdown()
    server2.shutdown()


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
    bookkeeper_bp = create_bookkeeper_blueprint()
    app.register_blueprint(bookkeeper_bp)
    app.roles.append(bookkeeper_bp.role)

    worker_bp, _ = create_worker_blueprint(up_dir, db_url, 1, analysis_func=dummy_analysis_func)
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


def test_client_delegate_workload(tmpdir):
    db_url = 'sqlite:///' + osp.join(tmpdir.strpath, "database.sqlite")
    user_client.evaluate_workload = Mock(return_value=0.0)
    dispatch_func, worker_pool, list_futures = user_client.get_job_dispathcher(db_url, 1, 320, 0,
                                                                               analysis_func=dummy_analysis_func)
    dispatch_func("dummy.avi")
    worker_pool.close()
    worker_pool.join()
    assert len(list_futures) == 1
    assert list_futures[0].get() == "dummy.csv"

    servers = create_bookkeeper_worker_servers(tmpdir)
    user_client.evaluate_workload = Mock(return_value=1.0)
    dispatch_func, worker_pool, list_futures = user_client.get_job_dispathcher(db_url, 1, 320, 0)
    with open("dummy.avi", "w") as f: f.write("nothing")
    dispatch_func("dummy.avi")
    assert len(list_futures) == 0

    shutdown_servers(servers)


def test_check_and_download(tmpdir):
    db_url = 'sqlite:///' + osp.join(tmpdir.strpath, "database.sqlite")
    user_client.evaluate_workload = Mock(return_value=1.0)
    dispatch_func, worker_pool, list_futures = user_client.get_job_dispathcher(db_url, 1, 320, 0,
                                                                               analysis_func=dummy_analysis_func)
    servers = create_bookkeeper_worker_servers(tmpdir)


    with open("dummy.avi", "w") as f: f.write("nothing")
    dispatch_func("dummy.avi")
    assert len(list_futures) == 0

    session = create_session(db_url)
    query = VideoStatuses.get_video_statuses(session)
    time.sleep(5)  # this should be enough in order to syncronize
    assert len(query) == 1
    VideoStatuses.remove_dead_requests(session)
    query = VideoStatuses.get_video_statuses(session)
    assert len(query) == 1

    shutdown_servers(servers)

