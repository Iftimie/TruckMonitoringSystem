import requests
from truckms.service_v2.api import P2PFlaskApp, validate_function_signature
from truckms.service.bookkeeper import create_bookkeeper_p2pblueprint
import multiprocessing
from flask import make_response, jsonify
from truckms.service_v2.api import derive_vars_from_function
import time
import os
from functools import wraps, partial
from truckms.service_v2.p2pdata import p2p_route_insert_one, deserialize_doc_from_net, p2p_route_pull_update_one, p2p_route_push_update_one
from truckms.service_v2.p2pdata import find, p2p_push_update_one, TinyMongoClientClean
from truckms.service_v2.registry_args import get_class_dictionary_from_func
import inspect
import logging
logger = logging.getLogger(__name__)


def call_remote_func(ip, port, db, col, func_name, identifier):
    res = requests.post(
        "http://{ip}:{port}/execute_function/{db}/{col}/{fname}/{identifier}".format(ip=ip, port=port, db=db,
                                                                                     col=col, fname=func_name,
                                                                                 identifier=identifier), files={},data={})
    return res


def check_remote_identifier(ip, port, db, col, func_name, identifier):
    res = requests.get(
        "http://{ip}:{port}/identifier_available/{db}/{col}/{fname}/{identifier}".format(ip=ip, port=port, db=db,
                                                                                     col=col, fname=func_name,
                                                                                 identifier=identifier), files={},data={})
    if res.status_code == 200 and res.content == b'yes':
        return True
    elif res.status_code == 404 and res.content == b'no':
        return False
    else:
        raise ValueError("Problem")


def function_executor(f, identifier, db, col, db_url, key_interpreter):
    kwargs_ = find(db_url, db, col, {"identifier": identifier}, key_interpreter)[0]
    kwargs = {k:kwargs_[k] for k in inspect.signature(f).parameters.keys()}

    logger.info("Executing function: " + f.__name__)
    update_ = f(**kwargs)
    update_['finished'] = True
    if not all(isinstance(k, str) for k in update_.keys()):
        raise ValueError("All keys in the returned dictionary must be strings in func {}".format(f.__name__))
    filter_ = {"identifier": identifier}
    p2p_push_update_one(db_url, db, col, filter_, update_)
    # TODO key interpreter might be necessary in p2p_push_update_one from the returned dictionary. This can be solved by annotation the function with {"key":"value"}
    return update_


def route_execute_function(identifier, f, db_url, db, col, key_interpreter, can_do_locally_func, self):
    """
    Function designed to be decorated with flask.app.route
    The function should be partially applied will all arguments

    # TODO identifier should be receives using a post request or query parameter or url parameter as in here
    #  most of the other routed functions are requesting post json

    Args from network:
        identifier: string will be received when there is a http call

    Args from partial application of the function:
        f: function to execute
        db_path, db, col are part of tinymongo db
        key_interpreter: dictionary containing keys and the expected data types
        can_do_locally_func: function that returns True or False and says if the current function should be executed or not
        self: P2PFlaskApp instance. this instance contains a worker pool and a list of futures #TODO maybe instead of self, only these arguments should be passed
    Returns:
        flask response that will contain the dictionary as a json in header metadata and one file (which may be an archive for multiple files)
    """
    if can_do_locally_func():
        new_f = wraps(f)(
            partial(function_executor,
                    f=f, identifier=identifier,
                    db_url=db_url, db=db, col=col,
                    key_interpreter=key_interpreter))
        res = self.worker_pool.apply_async(func=new_f)
        TinyMongoClientClean(db_url)[db][col].update_one({"identifier": identifier}, {"started": f.__name__})
        self.list_futures.append(res)

    return make_response("ok")


def search_work(db_url, db, collection, func_name, time_limit):

    col = list(TinyMongoClientClean(db_url)[db][collection].find({}))

    col = list(filter(lambda item: "finished" not in item, col))
    col1 = filter(lambda item: "started" not in item, col)
    col2 = filter(lambda item: "started" in item and item['started'] != func_name, col)
    col3 = filter(lambda item: "started" in item and item['started'] == func_name and (time.time() - item['timestamp']) > time_limit * 3600, col)

    col = list(col1) + list(col2) + list(col3)
    # list(col) + list(col2)
    # TODO fix this. it returns items that have finished
    if col:
        col.sort(key=lambda item: item["timestamp"])  # might allready be sorted
        item = col[0]
        TinyMongoClientClean(db_url)[db][collection].update_one({"identifier": item["identifier"]},
                                                                        {"started": func_name})
        return jsonify({"identifier": item["identifier"]})
    else:
        return jsonify({})


def route_identifier_available(db_path, db, col, identifier):
    """
    Function designed to be decorated with flask.app.route
    The function should be partially applied will all arguments

    The function will return a boolean about the received identifier already exists or not in the database

    Args from network:
        identifier: string will be received when there is a http call
    Args from partial application of the function:
        db_path, db, col are part of tinymongo db
    """
    collection = find(db_path, db, col, {"identifier": identifier})
    if len(collection) == 0:
        return make_response("yes", 200)
    else:
        return make_response("no", 404)


def register_p2p_func(self, cache_path, can_do_locally_func=lambda: True, time_limit=12):
    """
    In p2p brokerworker will have the role of either executing a function that was registered (worker role), or store the arguments in a
     database in order to execute the function later by a clientworker (broker role).

    Args:
        self: P2PFlaskApp object this instance is passed as argument from create_p2p_client_app. This is done like that
            just to avoid making redundant Classes. Just trying to make the code more functional
        cache_path: path to a directory that serves storing information about function calls in a database
        can_do_locally_func: function that returns True if work can be done locally and false if it should be done later
            by this current node or by a clientworker
        limit=hours
    """

    def inner_decorator(f):
        key_interpreter, db_url, db, col = derive_vars_from_function(f, cache_path)
        updir = os.path.join(cache_path, db, col)  # upload directory
        os.makedirs(updir, exist_ok=True)

        # these functions below make more sense in p2p_data.py
        p2p_route_insert_one_func = wraps(p2p_route_insert_one)(
            partial(p2p_route_insert_one,
                    db=db, col=col, db_path=db_url,
                    deserializer=partial(deserialize_doc_from_net, up_dir=updir),
                    key_interpreter=key_interpreter))
        self.route("/insert_one/{db}/{col}".format(db=db, col=col), methods=['POST'])(p2p_route_insert_one_func)

        p2p_route_push_update_one_func = wraps(p2p_route_push_update_one)(
            partial(p2p_route_push_update_one,
                    db_path=db_url, db=db, col=col,
                    deserializer=partial(deserialize_doc_from_net, up_dir=updir)))
        self.route("/push_update_one/{db}/{col}".format(db=db, col=col), methods=['POST'])(p2p_route_push_update_one_func)

        p2p_route_pull_update_one_func = wraps(p2p_route_pull_update_one)(
            partial(p2p_route_pull_update_one,
                    db_path=db_url, db=db, col=col))
        self.route("/pull_update_one/{db}/{col}".format(db=db, col=col), methods=['POST'])(p2p_route_pull_update_one_func)

        execute_function_partial = wraps(f)(
            partial(route_execute_function,
                    f=f, db_url=db_url, db=db, col=col,
                    key_interpreter=key_interpreter, can_do_locally_func=can_do_locally_func, self=self))
        self.route('/execute_function/{db}/{col}/{fname}/<identifier>'.format(db=db, col=col, fname=f.__name__), methods=['POST'])(execute_function_partial)

        search_work_partial = wraps(search_work)(
            partial(search_work,
                    db_url=db_url, db=db, collection=col,
                    func_name=f.__name__, time_limit=time_limit))
        self.route("/search_work/{db}/{col}/{fname}".format(db=db, col=col, fname=f.__name__), methods=['POST'])(search_work_partial)

        identifier_available_partial = wraps(route_identifier_available)(
            partial(route_identifier_available,
                    db_path=db_url, db=db, col=col))
        self.route("/identifier_available/{db}/{col}/{fname}/<identifier>".format(db=db, col=col, fname=f.__name__), methods=['GET'])(identifier_available_partial)

    return inner_decorator


def heartbeat(db_url, db="tms"):
    """
    Pottential vulnerability from flooding here
    """
    collection = TinyMongoClientClean(db_url)[db]["broker_heartbeats"]
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

    p2p_flask_app.roles.append("brokerworker")
    bookkeeper_bp = create_bookkeeper_p2pblueprint(local_port=p2p_flask_app.local_port, app_roles=p2p_flask_app.roles,
                                                   discovery_ips_file=discovery_ips_file)
    p2p_flask_app.register_blueprint(bookkeeper_bp)

    p2p_flask_app.register_p2p_func = partial(register_p2p_func, p2p_flask_app)
    p2p_flask_app.worker_pool = multiprocessing.Pool(2)
    p2p_flask_app.list_futures = []
    # TODO I need to create a time regular func for those requests that are old in order to execute them in broker

    return p2p_flask_app
