from bookkeeper.service import create_microservice
import requests

def test_create_microservice():


    app = create_microservice()

    client = app.test_client()
    # aau-rainsnow dataset. found on kaggle
    data = {'cevaa': 'cevaaadata'}

    res = client.post("/node_states", json=data)
    assert (res.status_code == 200) # 302 is found redirect

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

    server = ServerThread(app)
    server.start()
    requests.post('http://localhost:5000/node_states')
    server.shutdown()