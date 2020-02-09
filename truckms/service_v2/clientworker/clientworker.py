from truckms.service.worker.worker_client import get_available_brokers
import requests
import logging
import tinymongo
from truckms.service_v2.p2pdata import p2p_pull_update_one, deserialize_doc_from_net, p2p_push_update_one, p2p_insert_one
from functools import partial
import inspect
import collections


def do_work(up_dir, db_url, local_port, func, db, collection):
    """

    """

    required_positional_args = []
    for k, v in inspect.signature(func, follow_wrapped=False).parameters.items():
        if v.default == inspect._empty:
            required_positional_args.append(v.name)
    assert all(key not in required_positional_args for key in ['identifier', 'nodes'])

    res, broker_ip, broker_port = find_response_with_work(local_port, collection, func.__name__)
    filter_ = {"identifier": res['identifier']}
    local_data = dict()
    local_data.update(filter_)
    local_data.update({k: None for k in required_positional_args})
    p2p_insert_one(db_url, db, collection, local_data, [broker_ip+":"+str(broker_port)], do_upload=False)

    deserializer = partial(deserialize_doc_from_net, up_dir=up_dir)
    # TODO instead of hardcoding here the required keys. those keys could be inspected form the function declaration,
    #  or a dectoator should be used to help the framework to know which resources the funciton needs in order to be executed on worker and workerclient.
    #  also the return value of the analysis func should be a dictionary
    #  actually the question is which function should do the pull and which function should do the push
    #  I believe that the user that implements the analysis function should have no responsability of knowing about p2p_data
    #  I will curently keep it like that, but needs refactoring
    p2p_pull_update_one(db_url, db, collection, filter_, required_positional_args, deserializer, hint_file_keys=func.hint_args_are_files)

    local_data_after_update = list(tinymongo.TinyMongoClient(db_url)[db][collection].find(filter_))[0]
    kwargs = {k: v for k, v in local_data_after_update.items() if k in required_positional_args}
    update = func(**kwargs)
    p2p_push_update_one(db_url, db, collection, filter_, update)
