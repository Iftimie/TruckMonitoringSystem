from flask import Flask, make_response, request, jsonify
from collections import namedtuple
from flask import Blueprint
from functools import partial, wraps
import threading
from werkzeug.serving import make_server
import requests


NodeState = namedtuple('NodeState', ['ip', 'port', 'workload', 'hardware', 'nickname', 'node_type', 'email'])


def node_states(set_states):
    if request.method == 'POST':
        received_states = request.json
        received_states = set(NodeState(**content) for content in received_states)
        set_states.update(received_states)
        return make_response("done", 200)
    else:
        return jsonify(list(set_states))


def create_blueprint():
    bookkeeper_bp = Blueprint("bookkeeper_bp", __name__)
    set_states = set()  # TODO replace with a local syncronized database
    func = (wraps(node_states)(partial(node_states, set_states)))
    bookkeeper_bp.route("/node_states", methods=['POST', 'GET'])(func)
    return bookkeeper_bp


def create_microservice():
    app = Flask(__name__)
    app.register_blueprint(create_blueprint())

    return app


def find_workload():
    return 0


class ServerThread(threading.Thread):

    def __init__(self, app, host='127.0.0.1', port=5000, central_host=None, central_port=None):
        threading.Thread.__init__(self)
        self.host = host
        self.port = port
        self.srv = make_server(host, port, app)
        self.ctx = app.app_context()
        self.client = app.test_client()
        data = [{'ip': self.host, 'port': self.port, 'workload': find_workload(), 'hardware': "Nvidia GTX 960M Intel i7",
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