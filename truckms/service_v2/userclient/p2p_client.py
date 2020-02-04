from truckms.service_v2.api import P2PFlaskApp, validate_arguments, find_update_callables, validate_function_signature
from truckms.service.bookkeeper import create_bookkeeper_p2pblueprint
from functools import wraps
from truckms.service_v2.userclient.userclient import select_lru_worker
from functools import partial
from truckms.service_v2.p2pdata import p2p_push_update_one, p2p_insert_one, find
from truckms.service_v2.p2pdata import p2p_pull_update_one, deserialize_doc_from_net
from truckms.service_v2.registry_args import get_class_dictionary_from_func
import logging
from truckms.service_v2.api import self_is_reachable
from truckms.service_v2.brokerworker.p2p_brokerworker import call_remote_func, function_executor
import multiprocessing
import collections
import inspect
import io
import os
from truckms.service_v2.api import wait_until_online
import time
import threading
import deprecated
import inspect
import requests
logger = logging.getLogger(__name__)


def p2p_progress_hook(curidx, endidx):
    necessary_args = ['db', 'col', 'identifier', 'db_url']
    actual_args = dict()
    frame_infos = inspect.stack()[:]
    for frame in frame_infos:
        if frame.function == function_executor.__name__:
            f_locals = frame.frame.f_locals
            actual_args = {k:f_locals[k] for k in necessary_args}
            break

    update_ = {"progress": curidx/endidx * 100}
    filter_ = {"identifier": actual_args['identifier']}
    p2p_push_update_one(actual_args['db_url'], actual_args['db'], actual_args['col'], filter_, update_)


from truckms.service_v2.registry_args import kicomp





def get_remote_future(f, identifier, db_url, db, col, key_interpreter_dict):
    up_dir = os.path.join(db_url, db)
    if not os.path.exists(up_dir):
        os.mkdir(up_dir)
    item = find(db_url, db, col, {"identifier": identifier}, key_interpreter_dict)[0]
    expected_keys = inspect.signature(f).return_annotation
    expected_keys_list = list(expected_keys.keys())
    expected_keys_list.append("progress")
    if any(item[k] is None for k in expected_keys):
        hint_file_keys = [k for k, v in expected_keys.items() if v == io.IOBase]

        search_filter = {"$or": [{"identifier": identifier}, {"identifier": item['remote_identifier']}]}
        p2p_pull_update_one(db_url, db, col, search_filter, expected_keys_list,
                            deserializer=partial(deserialize_doc_from_net, up_dir=up_dir), hint_file_keys=hint_file_keys)

    item = find(db_url, db, col, {"identifier": identifier}, key_interpreter_dict)[0]
    item = {k: item[k] for k in expected_keys_list}
    return item


def get_local_future(f, identifier, db_url, db, col, key_interpreter_dict):
    expected_keys = inspect.signature(f).return_annotation
    expected_keys_list = list(expected_keys.keys())
    expected_keys_list.append("progress")

    item = find(db_url, db, col, {"identifier": identifier}, key_interpreter_dict)[0]
    item = {k:v for k, v in item.items() if k in expected_keys_list}
    return item


class Future:

    def __init__(self, get_future_func, max_waiting_time=3600*24):
        self.get_future_func = get_future_func
        self.max_waiting_time = max_waiting_time

    def get(self):
        item = self.get_future_func()
        count_time = 0
        wait_time = 4
        while any(item[k] is None for k in item):
            item = self.get_future_func()
            time.sleep(wait_time)
            count_time += wait_time
            if count_time > self.max_waiting_time:
                raise ValueError("Waiting time exceeded")
            logger.info("Not done yet " + str(item))
        return item


def get_expected_keys(f):
    expected_keys = inspect.signature(f).return_annotation
    expected_keys = {k:None for k in expected_keys}
    expected_keys['progress'] = 0
    return expected_keys


def create_future(f, identifier, db_url, db, col, key_interpreter):
    item = find(db_url, db, col, {"identifier": identifier})[0]
    if item['nodes'] or 'remote_identifier' in item:
        return Future(partial(get_remote_future, f, identifier, db_url, db, col, key_interpreter))
    else:
        return Future(partial(get_local_future, f, identifier, db_url, db, col, key_interpreter))


# def verify_kwargs(db_url, db, col, kwargs, key_interpreter_dict):
#     collection = find(db_url, db, col, {"identifier": kwargs['identifier']}, key_interpreter_dict)
#     if collection:
#         item = collection[0]
#         if not all(kicomp(kwargs[k]) == kicomp(item[k]) for k in kwargs):
#             raise ValueError("different arguments for same identifier are not allowed")


def identifier_seen(db_url, identifier, db, col, expected_keys, time_limit=24):
    collection = find(db_url, db, col, {"identifier": identifier})

    if collection:
        assert len(collection) == 1
        item = collection[0]
        if (time.time() - item['timestamp']) > time_limit * 3600 and any(item[k] is None for k in expected_keys):
            # TODO the entry should actually be deleted instead of letting it be overwritten
            #  for elegancy
            logger.info("Time limit exceeded for item with identifier: "+item['identifier'])
            return False
        else:
            return True
    else:
        return False

from truckms.service_v2.registry_args import hash_kwargs
from truckms.service_v2.registry_args import kicomp
from truckms.service_v2.p2pdata import find
def create_identifier(db_url, db, col, kwargs, key_interpreter_dict):
    identifier = hash_kwargs({k:v for k, v in kwargs.items() if k in key_interpreter_dict})
    identifier_original = identifier  # deepcopy

    count = 1
    while True:
        collection = find(db_url, db, col, {"identifier": identifier}, key_interpreter_dict)
        if len(collection) == 0:
            # we found an identifier that is not in DB
            return identifier
        elif len(collection) != 1:
            # we should find only one doc that has the same hash
            raise ValueError("Multiple documents for a hash")
        elif all(kicomp(kwargs[k]) == kicomp(item[k]) for item in collection for k in kwargs):
            # we found exactly 1 doc with the same hash and we must check that it has the same arguments
            return identifier
        else:
            # we found different arguments that produce the same hash so we must modify the hash determinastically
            identifier = identifier_original + str(count)
            count += 1
            if count > 100:
                raise ValueError("Too many hash collisions. Change the hash function")


from truckms.service_v2.brokerworker.p2p_brokerworker import check_remote_identifier
def create_remote_identifier(local_identifier, check_remote_identifier_args):
    original_local_identifier = local_identifier
    args = {k: v for k, v in check_remote_identifier_args.items()}
    count = 1
    while True:
        args['identifier'] = local_identifier
        if check_remote_identifier(**args):
            return local_identifier
        else:
            local_identifier = original_local_identifier + str(count)
            count += 1
            if count > 100:
                raise ValueError("Too many hash collisions. Change the hash function")


from truckms.service_v2.p2pdata import update_one
def register_p2p_func(self, cache_path, can_do_locally_func=lambda: False):
    """
    In p2p client the register decorator will have the role of deciding if the function should be executed remotely or
    locally. It will store the input in a collection. If the node is reachable, then data will be updated automatically,
    otherwise data will be updated at subsequent calls.

    The decorator will assign to the function a property that allows accessing the database and the calls.

    Args:
        db_url: path to db
        db: string for database name
        col: string for collection name
        can_do_locally_func: function that returns True if work can be done locally and false if it should be done remotely
            if not specified, then it means all calls should be done remotely
        limit=hours
    """

    db_url = cache_path
    db = "p2p"

    def inner_decorator(f):
        validate_function_signature(f)
        key_interpreter = get_class_dictionary_from_func(f)

        col = f.__name__
        updir = os.path.join(cache_path, db, col)
        os.makedirs(updir, exist_ok=True)

        @wraps(f)
        def wrap(*args, **kwargs):

            identifier = create_identifier(db_url, db, col, kwargs, key_interpreter)
            expected_keys = get_expected_keys(f)
            if identifier_seen(db_url, identifier, db, col, expected_keys):
                logger.info("Returning future that may already be precomputed")
                return create_future(f, identifier, db_url, db, col, key_interpreter)

            validate_arguments(f, args, kwargs)
            kwargs.update(expected_keys)
            kwargs['identifier'] = identifier

            lru_ip, lru_port = select_lru_worker(self.local_port)

            if can_do_locally_func() or lru_ip is None:
                nodes = []
                p2p_insert_one(db_url, db, col, kwargs, nodes)
                new_f = wraps(f)(partial(function_executor, f=f, identifier=kwargs['identifier'], db_url=db_url, db=db, col=col, key_interpreter=key_interpreter))
                res = self.worker_pool.apply_async(func=new_f)
                logger.info("Executing function locally")
            else:
                nodes = [str(lru_ip) + ":" + str(lru_port)]
                # TODO check if the item was allready sent for processing
                remote_identifier = create_remote_identifier(kwargs['identifier'], {"ip": lru_ip, "port": lru_port, "db": db, "col": col, "func_name": f.__name__})
                local_identifier = kwargs['identifier']
                kwargs['identifier'] = remote_identifier
                p2p_insert_one(db_url, db, col, kwargs, nodes, current_address_func=partial(self_is_reachable, self.local_port))
                update_one(db_url, db, col, {'identifier':remote_identifier}, {"remote_identifier": remote_identifier, "identifier": local_identifier}, upsert=False)
                call_remote_func(lru_ip, lru_port, db, col, f.__name__, remote_identifier)
                logger.info("Dispacthed function work to {},{}".format(lru_ip, lru_port))
            return create_future(f, identifier, db_url, db, col, key_interpreter)
        return wrap
    return inner_decorator


from werkzeug.serving import make_server
class ServerThread(threading.Thread):

    def __init__(self, app):
        threading.Thread.__init__(self)
        self.srv = make_server('0.0.0.0', app.local_port, app)
        self.app = app
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.app.start_time_regular_thread()
        self.srv.serve_forever()

    def shutdown(self):
        self.app.stop_time_regular_thread()
        self.srv.shutdown()


def wait_for_discovery(local_port):
    """
    Try at most max_trials times to connect to p2p network or until the list of node si not empty
    """
    max_trials = 3
    count = 0
    while count < max_trials:
        logger.info("Waiting for nodes")
        res = requests.get('http://localhost:{}/node_states'.format(local_port)).json()  # will get the data defined above
        if len(res) != 0:
            break
        count+=1
        time.sleep(5)


def create_p2p_client_app(discovery_ips_file=None, p2p_flask_app=None):
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
    # p2p_flask_app.list_futures = []

    p2p_flask_app.background_thread = ServerThread(p2p_flask_app)
    p2p_flask_app.background_thread.start()
    wait_until_online(p2p_flask_app.local_port)
    wait_for_discovery(p2p_flask_app.local_port)

    return p2p_flask_app

