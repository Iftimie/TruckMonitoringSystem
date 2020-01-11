from tests.truckms.service_v2.functions_to_test import func_return_dict, func_return_val, complex_func, complex_func2
from truckms.service_v2.userclient.p2p_client import create_p2p_client_app
from truckms.service_v2.brokerworker.p2p_brokerworker import create_p2p_brokerworker_app
from truckms.service_v2.p2pdata import find
import traceback
import os.path as osp
import multiprocessing
import tinymongo


def some_func(identifier, arg1, arg2, arg3) -> dict:
    ret_value = {"value": "{},{},{},{}".format(identifier, arg1, arg2, arg3)}
    return ret_value


def some_progress_func() -> dict:
    pass


def some_other_progress_func() -> dict:
    return {"a": "b"}


class p2papp:

    def __init__(self):
        self.local_port = 1000
        self.worker_pool = multiprocessing.Pool(1)
        self.list_futures = []


def test_register_p2p_func(tmpdir):
    db_url, db, col = osp.join(tmpdir, "dburl"), "a", "b"
    app = create_p2p_client_app()
    dec_func = app.register_p2p_func(db_url, db, col)(some_func)
    dec_func(identifier="a", arg1="c", arg2="d", arg3=3)

    # app.worker_pool.close()
    # app.worker_pool.join()
    print(app.list_futures[0].get())

    try:
        dec_func("a", arg1="c", arg2="d", arg3=3)
        assert False
    except:
        traceback.print_exc()
        assert True

    try:
        dec_func(identifier="a", arg1=["c"], arg2="d", arg3=3)
        assert False
    except:
        traceback.print_exc()
        assert True

    try:
        dec_func(identifier="a", arg1=tuple("c"), arg2="d", arg3=3)
        assert False
    except:
        traceback.print_exc()
        assert True

    try:
        dec_func(identifier="a", arg1="value_for_key_is_file", arg2="d", arg3=3)
        assert False
    except:
        traceback.print_exc()
        assert True

    try:
        dec_func(identifier="a", arg1="value_for_key_is_file", arg2="d", arg3=3)
        assert False
    except:
        traceback.print_exc()
        assert True

    file_path = osp.join(tmpdir, "file.txt")
    with open(file_path, "w") as f:
        f.write("data")
    try:
        dec_func(identifier="a", arg1="1", arg2=open(file_path, "r"), arg3=open(file_path, "r"))
        assert False
    except:
        traceback.print_exc()
        assert True

    try:
        dec_func(identifier="a", arg1="1", arg2=open(file_path, "r"), arg3=1)
        assert False
    except:
        traceback.print_exc()
        assert True

    try:
        data = open(file_path, "r")
        data.close()
        dec_func(identifier="a", arg1="1", arg2=data, arg3=1)
        assert True
    except:
        traceback.print_exc()
        assert False


def test_register_p2p_func2(tmpdir):
    file_path = osp.join(tmpdir, "file.txt")
    with open(file_path, "w") as f:
        f.write("data")

    db_url, db, col = osp.join(tmpdir, "dburl"), "db", "col"
    app = create_p2p_client_app()
    decorated_func = app.register_p2p_func(db_url, db, col)(complex_func)

    try:
        data = open(file_path, "r")
        data.close()
        decorated_func(identifier="x", int_arg=10, str_arg="str", file_arg=data, func_arg=func_return_val, func_arg_ret_dict=func_return_dict)
        assert True
        app.worker_pool.close()
        app.worker_pool.join()
        item = tinymongo.TinyMongoClient(db_url)[db][col].find({"identifier": "x"})[0]
        assert item['identifier'] == 'x'
        assert item['int_arg'] == 10
        assert item['str_arg'] == 'str'
        assert item['nodes'] == []
        assert item['current_address'] is None
        assert item['file_arg'] == file_path
        assert item['func_arg_ret_dict_key'] == 10
        assert item['val'] == 'x,10,str,10'
    except:
        traceback.print_exc()
        assert False


class DummyObject:
    pass


def test_interaction_with_p2p_broker(tmpdir):
    file_path = osp.join(tmpdir, "file.txt")
    with open(file_path, "w") as f:
        f.write("data")

    db_url, db, col = osp.join(tmpdir, "dburl"), "db", "col"
    db_url_remote = osp.join(tmpdir, "dburl_remote")
    app = create_p2p_client_app()
    decorated_func = app.register_p2p_func(db_url, db, col, can_do_locally_func=lambda :False)(complex_func2)

    broker_app = create_p2p_brokerworker_app()
    broker_app.register_p2p_func(db_url_remote, db, col, can_do_locally_func=lambda: True)(complex_func2)

    def post_func(url, **kwargs):
        data = dict()
        data.update(kwargs["files"])
        data.update(kwargs["data"])
        print(url)
        client = broker_app.test_client()
        return client.post('/'.join(url.split("/")[3:]), data=data)

    from truckms.service_v2.brokerworker import p2p_brokerworker
    from truckms.service_v2.userclient import p2p_client
    from truckms.service_v2 import p2pdata
    mocked_requests = DummyObject()
    mocked_requests.post = post_func
    p2p_brokerworker.requests = mocked_requests
    p2pdata.requests = mocked_requests
    p2p_client.self_is_reachable = lambda :None
    p2p_client.select_lru_worker = lambda port:("ip", "port")

    try:
        data = open(file_path, "rb")
        decorated_func(identifier="x", int_arg=10, str_arg="str", file_arg=data, func_arg=func_return_val, func_arg_ret_dict=func_return_dict)
        assert True
        broker_app.worker_pool.close()
        broker_app.worker_pool.join()

        item = find(db_url_remote,db, col, {})[0]
        assert item['identifier'] == 'x'
        assert item['int_arg'] == 10
        assert item['str_arg'] == 'str'
        assert item['nodes'] == []
        assert item['current_address'] is None
        assert item['file_arg'] ==osp.join(tmpdir, db_url_remote, db, col, 'file.txt')
        assert item['func_arg_ret_dict_key'] == 10
        assert item['val'] == 'x,10,str,10'
    except:
        traceback.print_exc()
        assert False