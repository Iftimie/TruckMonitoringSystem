from truckms.service_v2.api import P2PFlaskApp, validate_arguments, find_update_callables
from truckms.service.bookkeeper import create_bookkeeper_p2pblueprint
from functools import wraps
from truckms.service_v2.userclient.userclient import select_lru_worker
import inspect
from functools import partial
from truckms.service_v2.p2pdata import p2p_pull_update_one, default_deserialize, p2p_push_update_one, p2p_insert_one
import logging
from truckms.service_v2.api import self_is_reachable
from truckms.service_v2.brokerworker.brokerworker import call_remote_func
import multiprocessing
import requests
import collections
import io
logger = logging.getLogger(__name__)


def new_update_func(*args, identifier, db_url, db, col, func, **kwargs):
    """
    new_func must not be declared inside of decorate_p2p_update because it cannot be pickled
    """
    filter_ = {"identifier": identifier}
    update_ = func(*args, **kwargs)
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


def new_func(f, keys_file_paths, db_url, db, col, **cur_kwargs):
    """
    new_func must not be declared inside of process_func_kwargs_decorator because it cannot be pickled
    """
    for k in cur_kwargs:
        if cur_kwargs[k] == "value_for_key_is_file":
            file_handler = open(keys_file_paths[k], 'r')
            file_handler.close()
            cur_kwargs[k] = file_handler

    update_ = f(**cur_kwargs)
    # TODO actually this check might not be necessary
    if not all(isinstance(k, str) for k in update_.keys()):
        raise ValueError("All keys in the returned dictionary must be strings in func {}".format(f.__name__))

    filter_ = {"identifier": cur_kwargs["identifier"]}
    p2p_push_update_one(db_url, db, col, filter_, update_)

    return update_


def process_func_kwargs_decorator(f, kwargs, db_url, db, col):
    """
    File handlers cannot be serialized (although these are closed) when running the function f in a worker pool
    """
    new_kwargs = {k: v for k, v in kwargs.items()}
    keys_file_paths = dict()

    for k in new_kwargs:
        if isinstance(new_kwargs[k],  io.IOBase):
            keys_file_paths[k] = new_kwargs[k].name
            new_kwargs[k] = "value_for_key_is_file"
    # @wraps(f)
    # def new_func(**cur_kwargs):
    #     """only receives keyword arguments that are definitely not files, because these have been separated"""
    #     for k in cur_kwargs:
    #         if cur_kwargs[k] == "value_for_key_is_file":
    #             file_handler = open(keys_file_paths[k], 'r')
    #             file_handler.close()
    #             cur_kwargs[k] = file_handler
    #     return f(**cur_kwargs)
    decorated_func = wraps(f)(partial(new_func, f=f, keys_file_paths=keys_file_paths, db_url=db_url, db=db, col=col))
    return decorated_func, new_kwargs

def register_p2p_func(self, db_url, db, col, can_do_locally_func=lambda: False):
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
    """
    def inner_decorator(f):
        formal_args = list(inspect.signature(f).parameters.keys())
        if "identifier" not in formal_args:
            raise ValueError("In this p2p framework, one argument to the function must be the identifier. "
                             "This helps for memoization and retrieving the results from a function")

        if not ("return" in inspect.getsource(f) and inspect.signature(f).return_annotation == dict):
            raise ValueError("Function must return something. And must be return annotated with dict")

        @wraps(f)
        def wrap(*args, **kwargs):
            validate_arguments(args, kwargs)
            # this mongodb_data should be always before any mofication of the kwargs
            mongodb_data = {k: v for (k, v) in kwargs.items() if not isinstance(v, collections.Callable)}

            kwargs = decorate_update_callables(db_url, db, col, kwargs)
            new_f, kwargs = process_func_kwargs_decorator(f, kwargs, db_url, db, col)

            lru_ip, lru_port = select_lru_worker(self.local_port)


            if can_do_locally_func() or lru_ip is None:
                nodes = []
                p2p_insert_one(db_url, db, col, mongodb_data, nodes)
                res = self.worker_pool.apply_async(func=new_f, kwds=kwargs)
                self.list_futures.append(res)
                logger.info("Executing function locally")
            else:
                nodes = [str(lru_ip) + ":" + str(lru_port)]
                p2p_insert_one(db_url, db, col, mongodb_data, nodes, current_address_func=self_is_reachable)
                call_remote_func(lru_ip, lru_port, db, col, new_f.__name__, kwargs['identifier'])

                # TODO call remote func with identifier
                logger.info("Dispacthed function work to {},{}".format(lru_ip, lru_port))
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
    p2p_flask_app.list_futures = []

    return p2p_flask_app

