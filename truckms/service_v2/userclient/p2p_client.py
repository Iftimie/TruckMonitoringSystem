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


def verify_kwargs(db_url, db, col, kwargs, key_interpreter_dict):
    collection = find(db_url, db, col, {"identifier": kwargs['identifier']}, key_interpreter_dict)
    if collection:
        item = collection[0]
        if not all(kicomp(kwargs[k]) == kicomp(item[k]) for k in kwargs):
            raise ValueError("different arguments for same identifier are not allowed")


def identifier_seen(db_url, kwargs, db, col):
    collection = find(db_url, db, col, {"identifier": kwargs['identifier']})
    if collection:
        return True
    else:
        return False


def get_remote_future(f, kwargs, db_url, db, col, key_interpreter_dict):
    up_dir = os.path.join(db_url, db)
    if not os.path.exists(up_dir):
        os.mkdir(up_dir)
    item = find(db_url, db, col, {"identifier": kwargs['identifier']}, key_interpreter_dict)[0]
    expected_keys = inspect.signature(f).return_annotation
    expected_keys_list = list(expected_keys.keys())
    expected_keys_list.append("progress")
    if any(item[k] is None for k in expected_keys):
        hint_file_keys = [k for k, v in expected_keys.items() if v == io.IOBase]

        p2p_pull_update_one(db_url, db, col, {"identifier": kwargs['identifier']}, expected_keys_list,
                            deserializer=partial(deserialize_doc_from_net, up_dir=up_dir), hint_file_keys=hint_file_keys)

    item = find(db_url, db, col, {"identifier": kwargs['identifier']}, key_interpreter_dict)[0]
    item = {k: item[k] for k in expected_keys_list}
    return item


def get_local_future(f, kwargs, db_url, db, col, key_interpreter_dict):
    expected_keys = inspect.signature(f).return_annotation
    expected_keys_list = list(expected_keys.keys())
    expected_keys_list.append("progress")

    item = find(db_url, db, col, {"identifier": kwargs['identifier']}, key_interpreter_dict)[0]
    item = {k:v for k, v in item.items() if k in expected_keys_list}
    return item


class Future:

    def __init__(self, get_future_func):
        self.get_future_func = get_future_func

    def get(self):
        item = self.get_future_func()
        while any(item[k] is None for k in item):
            item = self.get_future_func()
            time.sleep(4)
            logger.info("Not done yet " + str(item))
        return item


def add_expected_keys(f, kwargs):
    expected_keys = inspect.signature(f).return_annotation
    for k in expected_keys:
        kwargs[k] = None
    kwargs['progress'] = 0
    return kwargs


def create_future(f, kwargs, db_url, db, col, key_interpreter):
    item = find(db_url, db, col, {"identifier": kwargs['identifier']})[0]
    if item['nodes']:
        return Future(partial(get_remote_future, f, kwargs, db_url, db, col, key_interpreter))
    else:
        return Future(partial(get_local_future, f, kwargs, db_url, db, col, key_interpreter))


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

            verify_kwargs(db_url, db, col, kwargs, key_interpreter)
            if identifier_seen(db_url, kwargs, db, col):
                logger.info("Returning precomputed output")
                return create_future(f, kwargs, db_url, db, col, key_interpreter)

            validate_arguments(f, args, kwargs)

            # kwargs = decorate_update_callables(db_url, db, col, kwargs)
            kwargs = add_expected_keys(f, kwargs)

            lru_ip, lru_port = select_lru_worker(self.local_port)

            if can_do_locally_func() or lru_ip is None:
                nodes = []
                p2p_insert_one(db_url, db, col, kwargs, nodes, key_interpreter)
                new_f = wraps(f)(partial(function_executor, f=f, identifier=kwargs['identifier'], db_url=db_url, db=db, col=col, key_interpreter=key_interpreter))
                res = self.worker_pool.apply_async(func=new_f)
                logger.info("Executing function locally")
            else:
                nodes = [str(lru_ip) + ":" + str(lru_port)]
                # TODO check if the item was allready sent for processing
                p2p_insert_one(db_url, db, col, kwargs, nodes, key_interpreter, current_address_func=partial(self_is_reachable, self.local_port))
                call_remote_func(lru_ip, lru_port, db, col, f.__name__, kwargs['identifier'])
                logger.info("Dispacthed function work to {},{}".format(lru_ip, lru_port))
            return create_future(f, kwargs, db_url, db, col, key_interpreter)
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

