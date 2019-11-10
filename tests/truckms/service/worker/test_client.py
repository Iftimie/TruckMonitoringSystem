from truckms.service.bookkeeper import ServerThread
from truckms.service.bookkeeper import create_microservice
from truckms.service import bookkeeper
import requests
from truckms.service.worker.client import select_lru_worker
from truckms.service.worker import client
from mock import Mock
import os.path as osp
from truckms.service import worker
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


def create_servers():
    r"""Look in tests\service\bookkeeper\test_servic.py for details about the communication"""
    bookkeeper.find_workload = Mock(return_value=50)
    server1 = ServerThread(create_microservice(), port=5000)
    server1.start()  # at ths step there is only one node state, and only server1 knows about it (about itself)

    bookkeeper.find_workload = Mock(return_value=100)
    server2 = ServerThread(create_microservice(), port=5001, central_host='127.0.0.1', central_port=5000)
    server2.start()  # at this step server 1 knows about itself and server 2
    # server 2 knows about itself and server 1
    bookkeeper.find_workload = Mock(return_value=0)
    server3 = ServerThread(create_microservice(), port=5002, central_host='127.0.0.1', central_port=5000)
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

class PickableMock(Mock):
    def __init__(self):
        super(Mock, self).__init__()
        self.return_value = "dummy.csv"

    def __reduce__(self):
        return (PickableMock, ())

def test_client_delegate_workload(tmpdir):
    db_url = 'sqlite:///' + osp.join(tmpdir.strpath, "database.sqlite")
    client.evaluate_workload = Mock(return_value=0.0)
    worker.client.analyze_movie = PickableMock()
    dispatch_func, worker_pool, list_futures = client.get_job_dispathcher(db_url, 1, 320, 0)
    dispatch_func("dummy.avi")
    worker_pool.close()
    worker_pool.join()
    assert len(list_futures) == 1
    assert list_futures[0].get() == "dummy.csv"

    servers = create_servers()
    client.evaluate_workload = Mock(return_value=1.0)
    dispatch_func, worker_pool, list_futures = client.get_job_dispathcher(db_url, 1, 320, 0)
    dispatch_func("dummy")
    assert len(list_futures) == 0
    shutdown_servers(servers)

