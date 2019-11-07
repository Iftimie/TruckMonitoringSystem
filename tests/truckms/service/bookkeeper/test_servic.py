from truckms.service.bookkeeper import create_microservice
import requests
from werkzeug.serving import make_server
import threading


class ServerThread(threading.Thread):

    def __init__(self, app, host='127.0.0.1', port=5000, central_host=None, central_port=None):
        threading.Thread.__init__(self)
        self.host = host
        self.port = port
        self.srv = make_server(host, port, app)
        self.ctx = app.app_context()
        self.client = app.test_client()
        data = [{'ip': self.host, 'port': self.port, 'workload': 0, 'hardware': "Nvidia GTX 960M Intel i7",
                 'nickname': "rmstn",
                 'node_type': "bookkeeper", 'email': 'iftimie.alexandru.florentin@gmail.com'}]
        # register self state to local service
        res = self.client.post("/node_states", json=data)
        if central_host is not None and central_port is not None:
            # register self state to central service
            res = requests.post('http://{}:{}/node_states'.format(central_host, central_port), json=data, timeout=1)
            # register remote states to local service
            res = self.client.post("/node_states", json=requests.get('http://{}:{}/node_states'.format(central_host, central_port)).json())

        self.ctx.push()

    def run(self):
        self.srv.serve_forever()

    def shutdown(self):
        self.srv.shutdown()


def test_method_of_service():
    app = create_microservice()

    current_port = 5000
    server = ServerThread(app, port=current_port)
    server.start()

    res = requests.get('http://localhost:5000/node_states').json()  # will get the data defined above
    assert res == [{'ip': '127.0.0.1', 'port': 5000, 'workload': 0, 'hardware': "Nvidia GTX 960M Intel i7", 'nickname': "rmstn",
                'node_type': "bookkeeper", 'email': 'iftimie.alexandru.florentin@gmail.com'}]
    server.shutdown()


def test_two_services():
    server1 = ServerThread(create_microservice(), port=5000)
    server1.start()
    server2 = ServerThread(create_microservice(), port=5001, central_host='127.0.0.1', central_port=5000)
    server2.start()

    res1 = requests.get('http://localhost:5000/node_states').json()  # will get the data defined above
    res2 = requests.get('http://localhost:5001/node_states').json()  # will get the data defined above
    print (res1)
    print (res2)
    assert len(res1) == 2
    assert len(res2) == 2
    assert res1 == res2
    server1.shutdown()
    server2.shutdown()
