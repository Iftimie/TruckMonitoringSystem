import requests
from truckms.service_v2.api import P2PFlaskApp, validate_function_signature
from truckms.service.bookkeeper import create_bookkeeper_p2pblueprint
from functools import partial
import multiprocessing
from flask import make_response
import tinymongo
import time
import os
from functools import wraps, partial
from truckms.service_v2.api import validate_arguments
from truckms.service_v2.p2pdata import p2p_route_insert_one, default_deserialize, p2p_route_pull_update_one, p2p_route_push_update_one

#  call_remote_func(lru_ip, lru_port, db, col, new_f, kwargs)
def call_remote_func(ip, port, db, col, func_name, identifier):
    requests.post(
        "http://{ip}:{port}/execute_function/{db}/{col}/{fname}/{identifier}".format(ip=ip, port=port, db=db,
                                                                                     col=col, fname=func_name,
                                                                                     identifier=identifier))

def nonlocal_wrap(f, **kwargs):
    print("asdasd")
    # tinimongo find identifier
    # deserialize functions
    pass


def register_p2p_func(self, db_url, db, col, can_do_locally_func=lambda: False, current_address_func=lambda: None):
    updir = os.path.join(db_url, db, col)
    os.makedirs(updir, exist_ok=True)
    def inner_decorator(f):
        validate_function_signature(f)

        wrap = wraps(f)(partial(nonlocal_wrap, f=f))

        # @wraps(f)
        # def wrap(*args, **kwargs):
        #     # validate_arguments(args, kwargs) # argument validation is allready done in p2p_client.py
        #     kwargs['identifier']
        #     pass

        # self_is_reachable
        new_deserializer = partial(default_deserialize, up_dir=updir)

        p2p_route_insert_one_func = (wraps(p2p_route_insert_one)(partial(p2p_route_insert_one, db_path=db_url, deserializer=new_deserializer,current_address_func=current_address_func)))
        self.route("/insert_one/{db}/{col}".format(db=db, col=col), methods=['POST'])(p2p_route_insert_one_func)
        p2p_route_push_update_one_func = (wraps(p2p_route_push_update_one)(partial(p2p_route_push_update_one, db_path=db_url, deserializer=new_deserializer)))
        self.route("/push_update_one/{db}/{col}".format(db=db, col=col), methods=['POST'])(p2p_route_push_update_one_func)
        p2p_route_pull_update_one_func = (wraps(p2p_route_pull_update_one)(partial(p2p_route_pull_update_one, db_path=db_url)))
        self.route("/pull_update_one/{db}/{col}".format(db=db, col=col), methods=['POST'])(p2p_route_pull_update_one_func)
        self.route('/execute_function/{db}/{col}/{fname}/<identifier>'.format(db=db, col=col, fname=f.__name__),
                   methods=['GET'])(wrap)

    return inner_decorator

def heartbeat(db_url, db="tms"):
    """
    Pottential vulnerability from flooding here
    """
    collection = tinymongo.TinyMongoClient(db_url)[db]["broker_heartbeats"]
    collection.insert_one({"time_of_heartbeat": time.time()})
    return make_response("Thank god you are alive", 200)


def create_p2p_brokerworker_app(discovery_ips_file=None, p2p_flask_app=None):
    """
    Returns a Flask derived object with additional features

    Args:
        port:
        discovery_ips_file: file with other nodes
    """
    if p2p_flask_app is None:
        p2p_flask_app = P2PFlaskApp(__name__)

    bookkeeper_bp = create_bookkeeper_p2pblueprint(local_port=p2p_flask_app.local_port, app_roles=p2p_flask_app.roles,
                                                   discovery_ips_file=discovery_ips_file)
    p2p_flask_app.register_blueprint(bookkeeper_bp)

    p2p_flask_app.register_p2p_func = partial(register_p2p_func, p2p_flask_app)
    p2p_flask_app.worker_pool = multiprocessing.Pool(1)
    p2p_flask_app.list_futures = []

    return p2p_flask_app
