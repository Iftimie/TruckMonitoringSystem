from truckms.service.bookkeeper import ServerThread
from truckms.service.bookkeeper import create_microservice
from truckms.service import bookkeeper
import requests
from truckms.service.worker.client import select_lru_worker
from mock import Mock


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
