from flask import make_response, request, jsonify
from truckms.service_v2.api import P2PFlaskApp, P2PBlueprint, find_free_port
from collections import namedtuple
from flask import Blueprint
from functools import partial, wraps
import threading
from werkzeug.serving import make_server
import requests
import logging
import traceback
from typing import Tuple
logger = logging.getLogger(__name__)
import socket
from typing import List


NodeState = namedtuple('NodeState', ['ip', 'port', 'workload', 'hardware', 'nickname', 'node_type', 'email'])
#TODO move this to a database. its too anoying to conver from dict to tuple, to namedtuple just to store them in a set...
# especially in def update_function(local_port, app_roles, discovery_ips_file)

def node_states(set_states):
    if request.method == 'POST':
        # TODO. this should not be ignored because a new node will not be able to publish it's state
        # if request.remote_addr != '127.0.0.1':
        #     return make_response("Just what do you think you're doing, Dave?", 403)
        received_states = request.json
        received_states = set(NodeState(**content) for content in received_states)
        set_states.update(received_states)
        return make_response("done", 200)
    else:
        # return jsonify(list(set_states))
        return jsonify([a._asdict() for a in set_states])


def create_bookkeeper_p2pblueprint(local_port: int, app_roles: List[str], discovery_ips_file: str) -> P2PBlueprint:
    """
    Creates the bookkeeper blueprint

    Args:
        local_port: integer
        app_roles:
        discovery_ips_file: path to file with initial configuration of the network. The file should contain a list with
            reachable addresses

    Return:
        P2PBluePrint
    """
    bookkeeper_bp = P2PBlueprint("bookkeeper_bp", __name__, role="bookkeeper")
    set_states = set()  # TODO replace with a local syncronized database
    func = (wraps(node_states)(partial(node_states, set_states)))
    bookkeeper_bp.route("/node_states", methods=['POST', 'GET'])(func)

    time_regular_func = partial(update_function, local_port, app_roles, discovery_ips_file)
    bookkeeper_bp.register_time_regular_func(time_regular_func)

    return bookkeeper_bp


def update_function(local_port, app_roles, discovery_ips_file):
    """
    Function for bookkeeper to make network discovery
    discovery_ips_file: can be None
    """
    try:
        res = requests.get('http://localhost:{}/node_states'.format(local_port)).json()  # will get the data defined above

        # own state
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_ = s.getsockname()[0]
        s.close()
        discovered_states = []
        discovered_states.append({'ip': ip_,
                              'port': local_port,
                              'workload': find_workload(),
                              'hardware': "Nvidia GTX 960M Intel i7",
                              'nickname': "rmstn",
                              'node_type': ",".join(app_roles),
                              'email': 'iftimie.alexandru.florentin@gmail.com'})

        try:
            externalipres = requests.get('http://checkip.dyndns.org/')
            part = externalipres.content.decode('utf-8').split(": ")[1]
            ip_ = part.split("<")[0]
            try:
                echo_response = requests.get('http://{}:{}/echo'.format(ip_, local_port), timeout=3)
                if echo_response.status_code == 200:
                    discovered_states.append({'ip': ip_,
                                              'port': local_port,
                                              'workload': find_workload(),
                                              'hardware': "Nvidia GTX 960M Intel i7",
                                              'nickname': "rmstn",
                                              'node_type': ",".join(app_roles),
                                              'email': 'iftimie.alexandru.florentin@gmail.com'})
            except:
                pass
        except:
            logger.info(traceback.format_exc())

        # other states
        if discovery_ips_file is not None:
            with open(discovery_ips_file, 'r') as f:
                for line in f.readlines():
                    ip, port, workload, hardware, nickname, nodetype,email = line.replace("\n", "").split(";")
                    port, workload = int(port), int(workload)

                    discovered_states.append({'ip': ip, 'port': port, 'workload': workload, 'hardware': hardware,
                                              'nickname': nickname,
                                              'node_type': nodetype.replace('"', ''), 'email': email})

        for state in res:
            try:
                discovered_ = requests.get('http://{}:{}/node_states'.format(state['ip'], state[
                    'port'])).json()  # TODO should rename everything from host to ip
                discovered_ = set(NodeState(**content) for content in discovered_)
                discovered_states += [item._asdict() for item in discovered_]
            except:
                #some adresses may be dead
                #TODO maybe remove them?
                pass

        discovered_states = set(NodeState(**content) for content in discovered_states)
        discovered_states = [item._asdict() for item in discovered_states]

        # also store them
        # TODO I should move this reading from here to the app creation and use app.test_client.get
        if discovery_ips_file is not None:
            with open(discovery_ips_file, 'w') as f:
                for state in discovered_states:
                    f.write("{ip};{port};{workload};{hardware};{nickname};{node_type};{email}\n".format(
                        ip=state['ip'], port=state['port'], workload=state['workload'], hardware=state['hardware'],
                        nickname=state['nickname'], node_type=state['node_type'], email=state['email']
                    ))

        # publish the results to the current node and also to the rest of the nodes
        requests.post('http://localhost:{}/node_states'.format(local_port), json=discovered_states)
        for state in res:
            try:
                response = requests.post('http://{}:{}/node_states'.format(state['ip'], state['port']), json=discovered_states)
            except:
                logger.info("{}{} no longer exists".format(state['ip'], state['port']))
                #some adresses may be dead
                #TODO maybe remove them?
                pass
    except:
        logger.info(traceback.format_exc())


def create_bookkeeper_service(local_port: int, discovery_ips_file: str) -> P2PFlaskApp:
    """
    Creates a bookkeeper service. The bookkeeper service has the role of discovering other nodes in the network.
    It consists of a server that handles GET or POST requests about the node states. It also has a background active
    function that makes requests to the local bookkeeper server and to remote bookkeeper servers in order to make new
    discoveries.

    Args:
        local_port: the local port tells the time_regular_func on which port to make requests to /node_states
        discovery_ips_file: path to a file containing node states. the network discovery starts from making requests to
            nodes found in this file

    Returns:
        P2PFlaskApp
    """
    app = P2PFlaskApp(__name__)
    bookkeeper_bp = create_bookkeeper_p2pblueprint(local_port, app.roles, discovery_ips_file)
    app.register_blueprint(bookkeeper_bp)

    return app


def find_workload():
    return 0


# class used only for test purposes
class ServerThread(threading.Thread):

    def __init__(self, app, host='0.0.0.0', port=5000, central_host=None, central_port=None):
        #TODO this ServerThread should actually call something from update_function, since the calls are similar

        threading.Thread.__init__(self)
        self.host = host
        self.port = port
        self.srv = make_server(host, port, app)
        self.ctx = app.app_context()
        self.client = app.test_client()
        assert self.host != 'localhost' and self.host != '127.0.0.1'
        # just to avoid confusion. localhost will be 127.0.0.1
        # I am not sure altough what 0.0.0.0 means
        # https://superuser.com/questions/949428/whats-the-difference-between-127-0-0-1-and-0-0-0-0
        # Typically you use bind-address 0.0.0.0 to allow connections from outside networks and sources. Many servers like MySQL typically bind to 127.0.0.1 allowing only loopback connections, requiring the admin to change it to 0.0.0.0 to enable outside connectivity.

        # update_function(local_port=port, app_roles=app.roles, discovery_ips_file=None)

        data = [{'ip': self.host, 'port': self.port, 'workload': find_workload(), 'hardware': "Nvidia GTX 960M Intel i7",
                 'nickname': "rmstn",
                 'node_type': ",".join(app.roles + ['bookkeeper']), 'email': 'iftimie.alexandru.florentin@gmail.com'}]
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