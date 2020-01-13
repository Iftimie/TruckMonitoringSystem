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
from truckms.service_v2.clientworker.clientworker import find_response_with_work
from truckms.service_v2.userclient.p2p_client import decorate_update_callables
logger = logging.getLogger(__name__)


def register_p2p_func(self, db_url, db, col):
    updir = os.path.join(db_url, db, col)
    os.makedirs(updir, exist_ok=True)

    def inner_decorator(f):

        validate_function_signature(f)
        key_interpreter = get_key_interpreter_by_signature(f)
        param_keys = list(inspect.signature(f).parameters.keys())
        key_return = list(inspect.signature(f).return_annotation.keys())
        hint_returned_file_keys = [k for k, v in inspect.signature(f).return_annotation.items() if v == io.IOBase]
        hint_args_file_keys = [k for k, v in inspect.signature(f).parameters.items() if v.annotation == io.IOBase]

        @wraps(f)
        def wrap(*agrs, **kwargs):
            res, broker_ip, broker_port = find_response_with_work(self.local_port, db, col, f.__name__)

            filter_ = {"identifier": res['identifier']}

            local_data = dict()
            local_data.update({k: None for k in param_keys})
            local_data.update({k: None for k in key_return})
            local_data["identifier"] = res['identifier']

            deserializer = partial(default_deserialize, up_dir=updir)
            p2p_insert_one(db_url, db, col, local_data, [broker_ip+":"+str(broker_port)], do_upload=False)
            p2p_pull_update_one(db_url, db, col, filter_, param_keys, deserializer, hint_file_keys=hint_args_file_keys)

            kwargs_ = find(db_url, db, col, filter_, key_interpreter)[0]
            kwargs = {k: kwargs_[k] for k in inspect.signature(f).parameters.keys()}
            #### THIS IS SOME CRAZY SHIT HACK THAT MAKES ME THINK THAT ITS BEETER THE DECORATED FUNCTION TO JUST ACCESS A FUNCTION
            for k in kwargs:
                if isinstance(kwargs[k], partial) and kwargs[k].func.__name__ == 'new_update_func':
                    kwargs[k].keywords['db_url'] = db_url

            update_ = f(**kwargs)
            update_['finished'] = True
            if not all(isinstance(k, str) for k in update_.keys()):
                raise ValueError("All keys in the returned dictionary must be strings in func {}".format(f.__name__))
            for k, v in update_.items():
                if isinstance(v, io.IOBase) and v.closed is True:
                    raise ValueError("Files returned must be opened")
            p2p_push_update_one(db_url, db, col, filter_, update_, key_interpreter)

        return wrap

    return inner_decorator


def create_p2p_clientworker_app(discovery_ips_file=None, p2p_flask_app=None):
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

    return p2p_flask_app