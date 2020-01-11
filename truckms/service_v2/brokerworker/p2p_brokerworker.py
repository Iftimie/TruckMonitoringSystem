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
from truckms.service_v2.p2pdata import get_key_interpreter_by_signature, find, p2p_push_update_one
import inspect

#  call_remote_func(lru_ip, lru_port, db, col, new_f, kwargs)
def call_remote_func(ip, port, db, col, func_name, identifier):
    res = requests.post(
        "http://{ip}:{port}/execute_function/{db}/{col}/{fname}/{identifier}".format(ip=ip, port=port, db=db,
                                                                                     col=col, fname=func_name,
                                                                                 identifier=identifier), files={},data={})
    return res


def function_executor(f, identifier, db, col, db_url, key_interpreter):
    kwargs_ = find(db_url, db, col, {"identifier": identifier}, key_interpreter)[0]
    kwargs = {k:kwargs_[k] for k in inspect.signature(f).parameters.keys()}

    #### THIS IS SOME CRAZY SHIT HACK THAT MAKES ME THINK THAT ITS BEETER THE DECORATED FUNCTION TO JUST ACCESS A FUNCTION
    for k in kwargs:
        if isinstance(kwargs[k], partial) and kwargs[k].func.__name__ == 'new_update_func':
            kwargs[k].keywords['db_url'] = db_url
    update_ = f(**kwargs)
    if not all(isinstance(k, str) for k in update_.keys()):
        raise ValueError("All keys in the returned dictionary must be strings in func {}".format(f.__name__))
    filter_ = {"identifier": identifier}
    p2p_push_update_one(db_url, db, col, filter_, update_)
    # TODO key interpreter might be necessary in p2p_push_update_one from the returned dictionary. This can be solved by annotation the function with {"key":"value"}
    return update_


def execute_function(identifier, f, db_url, db, col, key_interpreter, can_do_locally_func, self):
    if can_do_locally_func():
        new_f = wraps(f)(partial(function_executor, f=f, identifier=identifier, db_url=db_url, db=db, col=col, key_interpreter=key_interpreter))
        res = self.worker_pool.apply_async(func=new_f)
        self.list_futures.append(res)

    return make_response("ok")


def register_p2p_func(self, db_url, db, col, can_do_locally_func=lambda: True, current_address_func=lambda: None):
    updir = os.path.join(db_url, db, col)
    os.makedirs(updir, exist_ok=True)
    def inner_decorator(f):
        validate_function_signature(f)
        key_interpreter = get_key_interpreter_by_signature(f)

        execute_function_partial = wraps(f)(partial(execute_function, f=f, db_url=db_url, db=db, col=col, key_interpreter=key_interpreter, can_do_locally_func=can_do_locally_func, self=self))

        # self_is_reachable
        new_deserializer = partial(default_deserialize, up_dir=updir)

        # these functions below make more sense in p2p_data.py
        p2p_route_insert_one_func = (wraps(p2p_route_insert_one)(partial(p2p_route_insert_one, db=db, col=col, db_path=db_url,
                                                                         deserializer=new_deserializer,
                                                                         current_address_func=current_address_func,
                                                                         key_interpreter=key_interpreter)))
        self.route("/insert_one/{db}/{col}".format(db=db, col=col), methods=['POST'])(p2p_route_insert_one_func)
        p2p_route_push_update_one_func = (wraps(p2p_route_push_update_one)(partial(p2p_route_push_update_one, db_path=db_url, deserializer=new_deserializer)))
        self.route("/push_update_one/{db}/{col}".format(db=db, col=col), methods=['POST'])(p2p_route_push_update_one_func)
        p2p_route_pull_update_one_func = (wraps(p2p_route_pull_update_one)(partial(p2p_route_pull_update_one, db_path=db_url, db=db, col=col)))

        # TODO be careful with these paths as they have been changed form <> to {} (variable to fixed)
        self.route("/pull_update_one/{db}/{col}".format(db=db, col=col), methods=['POST'])(p2p_route_pull_update_one_func)
        self.route('/execute_function/{db}/{col}/{fname}/<identifier>'.format(db=db, col=col, fname=f.__name__),
                   methods=['POST'])(execute_function_partial)

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
    p2p_flask_app.worker_pool = multiprocessing.Pool(2)
    p2p_flask_app.list_futures = []

    return p2p_flask_app
