from truckms.service_v2.api import P2PFlaskApp, validate_arguments
from truckms.service.bookkeeper import create_bookkeeper_p2pblueprint
from functools import wraps
from truckms.service_v2.userclient.userclient import select_lru_worker
import inspect
from functools import partial
from truckms.service_v2.p2pdata import p2p_pull_update_one, default_deserialize, p2p_push_update_one, p2p_insert_one
import logging
from truckms.service_v2.api import self_is_reachable
from truckms.service_v2.brokerworker.brokerworker import call_remote_func
import requests
logger = logging.getLogger(__name__)


def register_p2p_func(self, db_url, db, col, can_do_locally_func=lambda : False):
    """
    In p2p client the register decorator will have the role of deciding if the function should be executed remotely or
    locally. It will store the input in a collection. If the node is reachable, then data will be updated automatically,
    otherwise data will be updated at subsequent calls.

    The decorator will assign to the function a property that allows accessing the database and the calls.

    Args:
        db_url: path to db
        db: string for database name
        col: string for collection name
        workload_evaluator: function that returns True if work can be done locally and false if it should be done remotely
            if not specified, then it means all calls should be done remotely
    """
    def inner_decorator(f):
        formal_args = list(inspect.signature(f).parameters.keys())
        if "identifier" not in formal_args:
            raise ValueError("In this p2p framework, one argument to the function must be the identifier. "
                             "This helps for memoization and retrieving the results from a function")
        @wraps(f)
        def wrap(*args, **kwargs):
            validate_arguments(args, kwargs)

            lru_ip, lru_port = select_lru_worker(self.local_port)

            data = kwargs

            if can_do_locally_func() or lru_ip is None:
                nodes = []
                p2p_insert_one(db_url, db, col, data, nodes)
                res = self.worker_pool.apply_async(func=f, kwds=kwargs)
                self.list_futures.append(res)
                logger.info("Executing function locally")
            else:
                nodes = [str(lru_ip) + ":" + str(lru_port)]
                p2p_insert_one(db_url, db, col, data, nodes, current_address_func=self_is_reachable)
                call_remote_func(lru_ip, lru_port, db, col, f.__name__, data['identifier'])

                # TODO call remote func with identifier
                logger.info("Dispacthed function work to {},{}".format(lru_ip, lru_port))


            return f(*args, **kwargs)
        return wrap
    return inner_decorator


def create_p2p_client_app(port, discovery_ips_file, *args, **kwargs):
    """
    Returns a Flask derived object with additional features

    Args:
        port:
        discovery_ips_file: file with other nodes
        *args: positional args for creating a flask app
        **kwargs: keyword args for creating a flask app
    """
    p2p_client_app = P2PFlaskApp(*args, **kwargs)
    bookkeeper_bp = create_bookkeeper_p2pblueprint(local_port=port, app_roles=p2p_client_app.roles,
                                                   discovery_ips_file=discovery_ips_file)
    p2p_client_app.register_blueprint(bookkeeper_bp)
    p2p_client_app.register_p2p_func = partial(register_p2p_func, p2p_client_app)

    return p2p_client_app

