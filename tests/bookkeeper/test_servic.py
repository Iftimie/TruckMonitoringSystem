from bookkeeper.service import create_microservice
import requests
from werkzeug.serving import make_server
import threading


class ServerThread(threading.Thread):

    def __init__(self, app, host='127.0.0.1', port=5000):
        threading.Thread.__init__(self)
        self.srv = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.srv.serve_forever()

    def shutdown(self):
        self.srv.shutdown()


def test_method_of_service():
    app = create_microservice(debug=True)

    current_port = 5000
    server = ServerThread(app, port=current_port)
    server.start()
    data = {'port': current_port,
            'workload': 0,
            'hardware': "Nvidia GTX 960M Intel i7",
            'nickname': "rmstn",
            'node_type': "worker",
            'email': 'iftimie.alexandru.florentin@gmail.com'}
    res = requests.post('http://localhost:5000/node_states', json=data)
    assert (res.status_code == 200)
    res = requests.get('http://localhost:5000/node_states').json()[0]  # will get the data defined above
    del res['ip']
    assert res == data
    server.shutdown()


def test_two_services():
    server1 = ServerThread(create_microservice(debug=True), port=5000)
    server2 = ServerThread(create_microservice(debug=True), port=5001)
    server1.start()
    server2.start()
    data = {'port': 5000, 'workload': 0, 'hardware': "Nvidia GTX 960M Intel i7", 'nickname': "rmstn",
            'node_type': "worker", 'email': 'iftimie.alexandru.florentin@gmail.com'}
    res = requests.post('http://localhost:5000/node_states', json=data)
    data = {'port': 5001, 'workload': 0, 'hardware': "Nvidia GTX 960M Intel i7", 'nickname': "rmstn",
            'node_type': "worker", 'email': 'iftimie.alexandru.florentin@gmail.com'}
    res = requests.post('http://localhost:5000/node_states', json=data)

    res = requests.get('http://localhost:5000/node_states').json()  # will get the data defined above
    assert len(res) == 2
    server1.shutdown()
    server2.shutdown()