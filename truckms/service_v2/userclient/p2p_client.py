from truckms.service_v2.api import P2PFlaskApp, validate_arguments, find_update_callables, validate_function_signature
from truckms.service.bookkeeper import create_bookkeeper_p2pblueprint
from functools import wraps
from truckms.service_v2.userclient.userclient import select_lru_worker
from functools import partial
from truckms.service_v2.p2pdata import p2p_push_update_one, p2p_insert_one, get_key_interpreter_by_signature, find
from truckms.service_v2.p2pdata import p2p_pull_update_one, default_deserialize
import logging
from truckms.service_v2.api import self_is_reachable
from truckms.service_v2.brokerworker.p2p_brokerworker import call_remote_func, function_executor
import multiprocessing
import collections
import inspect
import io
import os
import time
logger = logging.getLogger(__name__)


def new_update_func(*args, identifier, db_url, db, col, func, **kwargs):
    """
    new_func must not be declared inside of decorate_p2p_update because it cannot be pickled
    """
    filter_ = {"identifier": identifier}
    update_ = func(*args, **kwargs)
    update_['finished'] = True
    # TODO actually this check might not be necessary
    if not all(isinstance(k, str) for k in update_.keys()):
        raise ValueError("All keys in the returned dictionary must be strings in func {}".format(func.__name__))
    assert 'identifier' not in update_
    p2p_push_update_one(db_url, db, col, filter_, update_)
    return update_


def decorate_p2p_update(db_url, db, col, identifier, func):
    """
    Decorates a function that returns a dictionary. This dictionary will be stored in the collection.
    """

    # @wraps(func)
    # def decorated_func(*args, **kwargs):
    #     filter_ = {"identifier": identifier}
    #     update_ = func(*args, **kwargs)
    #     assert 'identifier' not in update_
    #     p2p_push_update_one(db_url, db, col, filter_, update_)
    #     return update_
    decorated_func = wraps(func)(partial(new_update_func, identifier=identifier, db_url=db_url, db=db, col=col, func=func))

    return decorated_func


def decorate_update_callables(db_url, db, col, kwargs):
    """
    For all functions that return a dictionary and are annotated with -> dict, will decorate them to store the returned
    dictionary into a collection
    """
    new_kwargs = {k: v for k, v in kwargs.items()}
    update_callables = find_update_callables(new_kwargs)
    for k, c in update_callables.items():
        new_kwargs[k] = decorate_p2p_update(db_url, db, col, identifier=new_kwargs['identifier'], func=c)
    return new_kwargs


# def new_func(f, db_url, db, col):
#     """
#     new_func must not be declared inside of process_func_kwargs_decorator because it cannot be pickled
#     """
#
#     for k in cur_kwargs:
#         if cur_kwargs[k] == "value_for_key_is_file":
#             file_handler = open(keys_file_paths[k], 'r')
#             file_handler.close()
#             cur_kwargs[k] = file_handler
#
#     update_ = f(**cur_kwargs)
#     # TODO actually this check might not be necessary
#     if not all(isinstance(k, str) for k in update_.keys()):
#         raise ValueError("All keys in the returned dictionary must be strings in func {}".format(f.__name__))
#
#     filter_ = {"identifier": cur_kwargs["identifier"]}
#     p2p_push_update_one(db_url, db, col, filter_, update_)
#
#     return update_
#
#
# def process_func_kwargs_decorator(f, identifier, db_url, db, col):
#     """
#     File handlers cannot be serialized (although these are closed) when running the function f in a worker pool
#     """
#     # new_kwargs = {k: v for k, v in kwargs.items()}
#     # keys_file_paths = dict()
#     #
#     # for k in new_kwargs:
#     #     if isinstance(new_kwargs[k],  io.IOBase):
#     #         keys_file_paths[k] = new_kwargs[k].name
#     #         new_kwargs[k] = "value_for_key_is_file"
#     # @wraps(f)
#     # def new_func(**cur_kwargs):
#     #     """only receives keyword arguments that are definitely not files, because these have been separated"""
#     #     for k in cur_kwargs:
#     #         if cur_kwargs[k] == "value_for_key_is_file":
#     #             file_handler = open(keys_file_paths[k], 'r')
#     #             file_handler.close()
#     #             cur_kwargs[k] = file_handler
#     #     return f(**cur_kwargs)
#     decorated_func = wraps(f)(partial(new_func, f=f, identifier=identifier, db_url=db_url, db=db, col=col))
#     return decorated_func


def compare_dicts(cur_kwargs, db_args):
    for k in cur_kwargs:
        if isinstance(cur_kwargs[k], collections.Callable):
            if isinstance(db_args[k], partial) and db_args[k].func.__name__ == 'new_update_func':
                # THIS IS FOR MY STUPID CASE OF FUNCTION THAT RETURNS DICT TO INSERT IN DATABASE
                assert inspect.signature(cur_kwargs[k]) == inspect.signature(db_args[k].keywords['func'])
            else:
                assert inspect.signature(cur_kwargs[k]) == inspect.signature(db_args[k])
        elif isinstance(cur_kwargs[k], io.IOBase):
            assert cur_kwargs[k].name == db_args[k].name
        else:
            assert cur_kwargs[k] == db_args[k]


def verify_kwargs(db_url, db, col, kwargs, key_interpreter_dict):
    collection = find(db_url, db, col, {"identifier": kwargs['identifier']}, key_interpreter_dict)
    if collection:
        item = collection[0]
        try:
            compare_dicts(kwargs, item)
        except:
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
    if any(item[k] is None for k in expected_keys):
        hint_file_keys = [k for k, v in expected_keys.items() if v == io.IOBase]
        p2p_pull_update_one(db_url, db, col, {"identifier": kwargs['identifier']}, list(expected_keys.keys()),
                            deserializer=partial(default_deserialize, up_dir=up_dir), hint_file_keys=hint_file_keys)

    item = find(db_url, db, col, {"identifier": kwargs['identifier']}, key_interpreter_dict)[0]
    item = {k: item[k] for k in expected_keys}
    return item


def get_local_future(f, kwargs, db_url, db, col, key_interpreter_dict):
    expected_keys = inspect.signature(f).return_annotation
    item = find(db_url, db, col, {"identifier": kwargs['identifier']}, key_interpreter_dict)[0]
    item = {k:v for k, v in item.items() if k in expected_keys}
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
    return kwargs


def create_future(f, kwargs, db_url, db, col, key_interpreter):
    item = find(db_url, db, col, {"identifier": kwargs['identifier']})[0]
    if item['nodes']:
        return Future(partial(get_remote_future, f, kwargs, db_url, db, col, key_interpreter))
    else:
        return Future(partial(get_local_future, f, kwargs, db_url, db, col, key_interpreter))


def register_p2p_func(self, cache_path, can_do_locally_func=lambda: False, limit=24):
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
        key_interpreter = get_key_interpreter_by_signature(f)

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

            kwargs = decorate_update_callables(db_url, db, col, kwargs)
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

    return p2p_flask_app

