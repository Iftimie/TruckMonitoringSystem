import requests
from truckms.service_v2.api import P2PFlaskApp
from truckms.service.bookkeeper import create_bookkeeper_p2pblueprint
import multiprocessing
from flask import make_response, jsonify
from truckms.service_v2.api import derive_vars_from_function
import time
import os
from flask import request
from functools import wraps, partial
from truckms.service_v2.p2pdata import p2p_route_insert_one, deserialize_doc_from_net, p2p_route_pull_update_one, \
    p2p_route_push_update_one
from truckms.service_v2.p2pdata import find, p2p_push_update_one, TinyMongoClientClean
import inspect
import logging
from json import dumps, loads
from truckms.service_v2.api import configure_logger
from collections import defaultdict
from passlib.hash import sha256_crypt


def call_remote_func(ip, port, db, col, func_name, filter, password):
    """
    Client wrapper function over REST Api call
    """
    data = {"filter_json": dumps(filter)}

    url = "http://{ip}:{port}/execute_function/{db}/{col}/{fname}".format(ip=ip, port=port, db=db, col=col,
                                                                          fname=func_name)
    res = requests.post(url, files={}, data=data, headers={"Authorization": password})
    return res


def check_remote_identifier(ip, port, db, col, func_name, identifier, password):
    url = "http://{ip}:{port}/identifier_available/{db}/{col}/{fname}/{identifier}".format(ip=ip, port=port, db=db,
                                                                                     col=col, fname=func_name,
                                                                                 identifier=identifier)
    res = requests.get(url, files={}, data={}, headers={"Authorization": password})
    if res.status_code == 200 and res.content == b'yes':
        return True
    elif res.status_code == 404 and res.content == b'no':
        return False
    else:
        raise ValueError("Problem")


def function_executor(f, filter, db, col, db_url, key_interpreter, logging_queue, password):
    qh = logging.handlers.QueueHandler(logging_queue)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers = []
    root.addHandler(qh)

    logger = logging.getLogger(__name__)

    kwargs_ = find(db_url, db, col, filter, key_interpreter)[0]
    kwargs = {k: kwargs_[k] for k in inspect.signature(f).parameters.keys()}

    logger.info("Executing function: " + f.__name__)
    try:
        update_ = f(**kwargs)
    except Exception as e:
        logger.error("Function execution crashed for filter: {}".format(str(filter)), exc_info=True)
        raise e
    logger.info("Finished executing function: " + f.__name__)
    update_['finished'] = True
    if not all(isinstance(k, str) for k in update_.keys()):
        raise ValueError("All keys in the returned dictionary must be strings in func {}".format(f.__name__))
    p2p_push_update_one(db_url, db, col, filter, update_, password)
    return update_


def route_execute_function(f, db_url, db, col, key_interpreter, can_do_locally_func, self):
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
    logger = logging.getLogger(__name__)

    filter = loads(request.form['filter_json'])

    if can_do_locally_func():
        new_f = wraps(f)(
            partial(function_executor,
                    f=f, filter=filter,
                    db_url=db_url, db=db, col=col,
                    key_interpreter=key_interpreter,
                    logging_queue=self._logging_queue,
                    password=self.crypt_pass))
        res = self.worker_pool.apply_async(func=new_f)
        TinyMongoClientClean(db_url)[db][col].update_one(filter, {"started": f.__name__})
        self.list_futures.append(res)
    else:
        logger.info("Cannot execute function now: " + f.__name__)

    return make_response("ok")


def route_search_work(db_url, db, collection, func_name, time_limit):
    logger = logging.getLogger(__name__)

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
        filter_ = {"identifier": item["identifier"], "remote_identifier": item['remote_identifier']}
        logger.info("Node{}(possible client worker) will ask for filter: {}".format(request.remote_addr, str(filter_)))
        TinyMongoClientClean(db_url)[db][collection].update_one(filter_, {"started": func_name})
        return jsonify({"filter": filter_})
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


def register_p2p_func(self, can_do_locally_func=lambda: True, time_limit=12):
    """
    In p2p brokerworker, this decorator will have the role of either executing a function that was registered (worker role), or store the arguments in a
     database in order to execute the function later by a clientworker (broker role).

    Args:
        self: P2PFlaskApp object this instance is passed as argument from create_p2p_client_app. This is done like that
            just to avoid making redundant Classes. Just trying to make the code more functional
        can_do_locally_func: function that returns True if work can be done locally and false if it should be done later
            by this current node or by a clientworker
        limit=hours
    """

    def inner_decorator(f):
        if f.__name__ in self.registry_functions:
            raise ValueError("Function name already registered")
        key_interpreter, db_url, db, col = derive_vars_from_function(f, self.cache_path)

        self.registry_functions[f.__name__]['key_interpreter'] = key_interpreter

        updir = os.path.join(self.cache_path, db, col)  # upload directory
        os.makedirs(updir, exist_ok=True)

        # these functions below make more sense in p2p_data.py
        p2p_route_insert_one_func = wraps(p2p_route_insert_one)(
            partial(self.pass_req_dec(p2p_route_insert_one),
                    db=db, col=col, db_path=db_url,
                    deserializer=partial(deserialize_doc_from_net, up_dir=updir, key_interpreter=key_interpreter)))

        self.route("/insert_one/{db}/{col}".format(db=db, col=col), methods=['POST'])(p2p_route_insert_one_func)

        p2p_route_push_update_one_func = wraps(p2p_route_push_update_one)(
            partial(self.pass_req_dec(p2p_route_push_update_one),
                    db_path=db_url, db=db, col=col,
                    deserializer=partial(deserialize_doc_from_net, up_dir=updir, key_interpreter=key_interpreter)))
        self.route("/push_update_one/{db}/{col}".format(db=db, col=col), methods=['POST'])(p2p_route_push_update_one_func)

        p2p_route_pull_update_one_func = wraps(p2p_route_pull_update_one)(
            partial(self.pass_req_dec(p2p_route_pull_update_one),
                    db_path=db_url, db=db, col=col))
        self.route("/pull_update_one/{db}/{col}".format(db=db, col=col), methods=['POST'])(p2p_route_pull_update_one_func)

        execute_function_partial = wraps(f)(
            partial(self.pass_req_dec(route_execute_function),
                    f=f, db_url=db_url, db=db, col=col,
                    key_interpreter=key_interpreter, can_do_locally_func=can_do_locally_func, self=self))
        self.route('/execute_function/{db}/{col}/{fname}'.format(db=db, col=col, fname=f.__name__), methods=['POST'])(execute_function_partial)

        search_work_partial = wraps(route_search_work)(
            partial(self.pass_req_dec(route_search_work),
                    db_url=db_url, db=db, collection=col,
                    func_name=f.__name__, time_limit=time_limit))
        self.route("/search_work/{db}/{col}/{fname}".format(db=db, col=col, fname=f.__name__), methods=['POST'])(search_work_partial)

        identifier_available_partial = wraps(route_identifier_available)(
            partial(self.pass_req_dec(route_identifier_available),
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


from truckms.service_v2.p2pdata import deserialize_doc_from_db
from truckms.service_v2.registry_args import remove_values_from_doc
def delete_old_finished_requests(cache_path, registry_functions, time_limit=24):
    db_url = cache_path
    db = TinyMongoClientClean(db_url)['p2p']
    collection_names = db.tinydb.tables() - {"_default"}
    for col_name in collection_names:
        key_interpreter_dict = registry_functions[col_name]['key_interpreter']

        col_items = list(db[col_name].find({}))
        col_items = filter(lambda item: "finished" in item, col_items)
        col_items = filter(lambda item: (time.time() - item['timestamp']) > time_limit * 3600, col_items)

        for item in col_items:
            if (time.time() - item['timestamp']) > time_limit * 3600:
                document = deserialize_doc_from_db(item, key_interpreter_dict)
                remove_values_from_doc(document)
                db[col_name].remove(item)

from truckms.service_v2.p2pdata import password_required
def create_p2p_brokerworker_app(discovery_ips_file=None, local_port=None, password="", cache_path=None):
    """
    Returns a Flask derived object with additional features

    Args:
        port:
        discovery_ips_file: file with other nodes
        cache_path: path to a directory that serves storing information about function calls in a database
    """
    configure_logger("brokerworker", module_level_list=[(__name__, 'INFO')])

    p2p_flask_app = P2PFlaskApp(__name__, local_port=local_port)
    p2p_flask_app.cache_path = cache_path

    p2p_flask_app.roles.append("brokerworker")
    bookkeeper_bp = create_bookkeeper_p2pblueprint(local_port=p2p_flask_app.local_port, app_roles=p2p_flask_app.roles,
                                                   discovery_ips_file=discovery_ips_file, db_url=cache_path)
    p2p_flask_app.register_blueprint(bookkeeper_bp)

    p2p_flask_app.registry_functions = defaultdict(dict)
    p2p_flask_app.register_p2p_func = partial(register_p2p_func, p2p_flask_app)
    p2p_flask_app.worker_pool = multiprocessing.Pool(2)
    p2p_flask_app.list_futures = []
    p2p_flask_app.pass_req_dec = password_required(password)
    p2p_flask_app.register_time_regular_func(partial(delete_old_finished_requests,
                                                     cache_path=cache_path,
                                                     registry_functions=p2p_flask_app.registry_functions))
    p2p_flask_app.crypt_pass = sha256_crypt.encrypt(password)
    # TODO I need to create a time regular func for those requests that are old in order to execute them in broker

    return p2p_flask_app
