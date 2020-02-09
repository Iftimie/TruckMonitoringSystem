from truckms.service_v2.api import P2PFlaskApp
from truckms.service.bookkeeper import create_bookkeeper_p2pblueprint
from functools import wraps
from functools import partial
from truckms.service_v2.p2pdata import p2p_insert_one
from truckms.service_v2.p2pdata import p2p_pull_update_one, deserialize_doc_from_net
import inspect
import io
import os
from truckms.service_v2.clientworker.clientworker import find_response_with_work
from truckms.service_v2.brokerworker.p2p_brokerworker import function_executor
from truckms.service_v2.api import derive_vars_from_function
from truckms.service_v2.api import configure_logger


def register_p2p_func(self, cache_path):
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

            _ = function_executor(f, filter_, db, col, db_url, key_interpreter, self._logging_queue)

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
    configure_logger("clientworker", module_level_list=[(__name__, 'INFO')])

    p2p_flask_app = P2PFlaskApp(__name__, local_port=local_port)

    bookkeeper_bp = create_bookkeeper_p2pblueprint(local_port=p2p_flask_app.local_port, app_roles=p2p_flask_app.roles,
                                                   discovery_ips_file=discovery_ips_file)
    p2p_flask_app.register_blueprint(bookkeeper_bp)
    p2p_flask_app.register_p2p_func = partial(register_p2p_func, p2p_flask_app)

    return p2p_flask_app