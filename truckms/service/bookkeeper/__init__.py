from flask import Flask, make_response, request, jsonify
from collections import namedtuple
from flask import Blueprint
from functools import partial, wraps
import threading
from werkzeug.serving import make_server
import requests
import logging
import traceback
logger = logging.getLogger(__name__)
import socket

NodeState = namedtuple('NodeState', ['ip', 'port', 'workload', 'hardware', 'nickname', 'node_type', 'email'])
#TODO move this to a database. its too anoying to conver from dict to tuple, to namedtuple just to store them in a set...
# especially in def update_function(local_port, app_roles, discovery_ips_file)

def node_states(set_states):
    if request.method == 'POST':
        if request.remote_addr != '127.0.0.1':
            return make_response("Just what do you think you're doing, Dave?", 403)
        received_states = request.json
        received_states = set(NodeState(**content) for content in received_states)
        set_states.update(received_states)
        return make_response("done", 200)
    else:
        return jsonify(list(set_states))


def create_bookkeeper_blueprint():
    bookkeeper_bp = Blueprint("bookkeeper_bp", __name__)
    set_states = set()  # TODO replace with a local syncronized database
    func = (wraps(node_states)(partial(node_states, set_states)))
    bookkeeper_bp.route("/node_states", methods=['POST', 'GET'])(func)
    bookkeeper_bp.role = "bookkeeper"
    return bookkeeper_bp


def update_function(local_port, app_roles, discovery_ips_file):
    """
    Function for bookkeeper to make network discovery
    """
    try:
        res = requests.get('http://localhost:{}/node_states'.format(local_port)).json()  # will get the data defined above
        res = set(NodeState(*content) for content in res)
        res = [item._asdict() for item in res]

        # own state
        discovered_states = [{'ip': socket.gethostbyname(socket.gethostname()),
                              'port': local_port,
                              'workload': find_workload(),
                              'hardware': "Nvidia GTX 960M Intel i7",
                              'nickname': "rmstn",
                              'node_type': ",".join(app_roles + ['bookkeeper']),
                              'email': 'iftimie.alexandru.florentin@gmail.com'}]

        # other states
        with open(discovery_ips_file, 'r') as f:
            ip, port, workload, hardware, nickname, nodetype,email = f.readline().split(";")
            discovered_states.append({'ip': ip, 'port': port, 'workload': workload, 'hardware': hardware,
                                      'nickname': nickname,
                                      'node_type': nodetype.replace('"', ''), 'email': email})

        for state in res:
            try:
                discovered_ = requests.get('http://{}:{}/node_states'.format(state['ip'], state[
                    'port'])).json()  # TODO should rename everything from host to ip
                discovered_ = set(NodeState(*content) for content in discovered_)
                discovered_states += [item._asdict() for item in discovered_]
            except:
                #some adresses may be dead
                pass

        # also store them
        with open(discovery_ips_file, 'w') as f:
            for state in discovered_states:
                f.write("{ip};{port};{workload};{hardware};{nickname};{node_type};{email}\n".format(
                    ip=state['ip'], port=state['port'], workload=state['workload'], hardware=state['hardware'],
                    nickname=state['nickname'], node_type=state['node_type'], email=state['email']
                ))



        requests.post('http://localhost:{}/node_states'.format(local_port), json=discovered_states)
    except:
        logger.info(traceback.format_exc())


def create_bookkeeper_app(local_port, app_roles, discovery_ips_file):
    bookkeeper_bp = create_bookkeeper_blueprint()
    time_regular_func = partial(update_function, local_port, app_roles, discovery_ips_file)
    return bookkeeper_bp, time_regular_func


def create_microservice():
    app = Flask(__name__)
    app.roles = []
    bookkeeper_bp = create_bookkeeper_blueprint()
    app.register_blueprint(bookkeeper_bp)
    app.roles.append(bookkeeper_bp.role)

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
        assert self.host != 'localhost' and self.host != '0.0.0.0'
        # just to avoid confusion. localhost will be 127.0.0.1
        # I am not sure altough what 0.0.0.0 means
        # https://superuser.com/questions/949428/whats-the-difference-between-127-0-0-1-and-0-0-0-0
        # Typically you use bind-address 0.0.0.0 to allow connections from outside networks and sources. Many servers like MySQL typically bind to 127.0.0.1 allowing only loopback connections, requiring the admin to change it to 0.0.0.0 to enable outside connectivity.
        data = [{'ip': self.host, 'port': self.port, 'workload': find_workload(), 'hardware': "Nvidia GTX 960M Intel i7",
                 'nickname': "rmstn",
                 'node_type': ",".join(app.roles), 'email': 'iftimie.alexandru.florentin@gmail.com'}]
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