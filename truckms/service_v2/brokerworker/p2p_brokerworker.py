import requests
from truckms.service_v2.api import P2PFlaskApp
from truckms.service.bookkeeper import create_bookkeeper_p2pblueprint
from functools import partial
import multiprocessing
from flask import make_response
import tinymongo
import time

#  call_remote_func(lru_ip, lru_port, db, col, new_f, kwargs)
def call_remote_func(ip, port, db, col, func_name, identifier):
    requests.post(
        "http://{ip}:{port}/execute_function/{db}/{col}/{fname}/{identifier}".format(ip=ip, port=port, db=db,
                                                                                     col=col, fname=func_name,
                                                                                     identifier=identifier))


def register_p2p_func(self, db_url, db, col, can_do_locally_func=lambda: False):
    pass


def heartbeat(db_url, db="tms"):
    """
    Pottential vulnerability from flooding here
    """
    collection = tinymongo.TinyMongoClient(db_url)[db]["broker_heartbeats"]
    collection.insert_one({"time_of_heartbeat": time.time()})
    return make_response("Thank god you are alive", 200)


def create_p2p_brokerworker_app(up_dir, discovery_ips_file=None, p2p_flask_app=None):
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
