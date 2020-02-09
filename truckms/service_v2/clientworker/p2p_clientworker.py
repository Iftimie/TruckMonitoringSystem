from truckms.service_v2.api import P2PFlaskApp
from truckms.service.bookkeeper import create_bookkeeper_p2pblueprint
from functools import wraps
from functools import partial
from truckms.service_v2.p2pdata import p2p_insert_one
from truckms.service_v2.p2pdata import p2p_pull_update_one, deserialize_doc_from_net
import inspect
import io
import os
from truckms.service_v2.brokerworker.p2p_brokerworker import function_executor
from truckms.service_v2.brokerworker import p2p_brokerworker
from truckms.service_v2.api import derive_vars_from_function
from truckms.service_v2.api import configure_logger
import logging
from truckms.service.worker.worker_client import get_available_brokers
import requests
import collections
import multiprocessing


def find_response_with_work(local_port, db, collection, func_name):
    logger = logging.getLogger(__name__)

    res_broker_ip = None
    res_broker_port = None
    res_json = dict()

    brokers = get_available_brokers(local_port=local_port)

    if not brokers:
        logger.info("No broker found")

    for broker in brokers:
        broker_ip, broker_port = broker['ip'], broker['port']
        try:
            res = requests.post('http://{}:{}/search_work/{}/{}/{}'.format(broker_ip, broker_port, db, collection, func_name), timeout=5)
            if isinstance(res.json, collections.Callable):
                returned_json = res.json()  # from requests lib
            else: # is property
                returned_json = res.json  # from Flask test client
            if returned_json and 'filter' in returned_json:
                logger.info("Found work from {}, {}".format(broker_ip, broker_port))
                res_broker_ip = broker_ip
                res_broker_port = broker_port
                res_json = returned_json
                break
        except:  # except connection timeout or something like that
            logger.info("broker unavailable {}:{}".format(broker_ip, broker_port))
            pass

    if res_broker_ip is None:
        logger.info("No work found")

    # TODO it may be possible that res allready contains broker ip and port?
    return res_json, res_broker_ip, res_broker_port


def register_p2p_func(self, cache_path, can_do_work_func):
    """
    In p2p clientworker, this decorator will have the role of deciding making a node behind a firewall or NAT capable of
    executing a function that receives input arguments from over the network.

    Args:
        self: P2PFlaskApp object this instance is passed as argument from create_p2p_client_app. This is done like that
            just to avoid making redundant Classes. Just trying to make the code more functional
        cache_path: path to a directory that serves storing information about function calls in a database
    """

    def inner_decorator(f):
        key_interpreter, db_url, db, col = derive_vars_from_function(f, cache_path)
        updir = os.path.join(cache_path, db, col)  # upload directory
        os.makedirs(updir, exist_ok=True)

        param_keys = list(inspect.signature(f).parameters.keys())
        key_return = list(inspect.signature(f).return_annotation.keys())
        hint_args_file_keys = [k for k, v in inspect.signature(f).parameters.items() if v.annotation == io.IOBase]

        @wraps(f)
        def wrap():
            if not can_do_work_func():
                return

            logger = logging.getLogger(__name__)
            logger.info("Searching for work")
            res, broker_ip, broker_port = find_response_with_work(self.local_port, db, col, f.__name__)
            if broker_ip is None:
                return

            filter_ = res['filter']

            local_data = {k: v for k, v in filter_.items()}
            local_data.update({k: None for k in param_keys})
            local_data.update({k: None for k in key_return})

            deserializer = partial(deserialize_doc_from_net, up_dir=updir, key_interpreter=key_interpreter)
            p2p_insert_one(db_url, db, col, local_data, [broker_ip + ":" + str(broker_port)], do_upload=False)
            p2p_pull_update_one(db_url, db, col, filter_, param_keys, deserializer, hint_file_keys=hint_args_file_keys)

            new_f = wraps(f)(
                partial(function_executor,
                        f=f, filter=filter_,
                        db_url=db_url, db=db, col=col,
                        key_interpreter=key_interpreter,
                        logging_queue=self._logging_queue))
            res = self.worker_pool.apply_async(func=new_f)
            self.list_futures.append(res)
            # _ = function_executor(f, filter_, db, col, db_url, key_interpreter, self._logging_queue)
            # something weird was happening with logging when the function was executed in the same thread

        self.register_time_regular_func(wrap)
        return None

    return inner_decorator


def create_p2p_clientworker_app(discovery_ips_file=None, local_port=None):
    """
    Returns a Flask derived object with additional features

    Args:
        port:
        discovery_ips_file: file with other nodes
    """
    configure_logger("clientworker", module_level_list=[(__name__, 'INFO'),
                                                        (p2p_brokerworker.__name__, 'INFO')])

    p2p_flask_app = P2PFlaskApp(__name__, local_port=local_port)

    bookkeeper_bp = create_bookkeeper_p2pblueprint(local_port=p2p_flask_app.local_port, app_roles=p2p_flask_app.roles,
                                                   discovery_ips_file=discovery_ips_file)
    p2p_flask_app.register_blueprint(bookkeeper_bp)
    p2p_flask_app.register_p2p_func = partial(register_p2p_func, p2p_flask_app)
    p2p_flask_app.worker_pool = multiprocessing.Pool(2)
    p2p_flask_app.list_futures = []

    return p2p_flask_app