from truckms.service_v2.clientworker.p2p_clientworker import create_p2p_clientworker_app
from truckms.service_v2.brokerworker.p2p_brokerworker import create_p2p_brokerworker_app
from truckms.service_v2.userclient.p2p_client import create_p2p_client_app
from truckms.service_v2.p2pdata import find
from tests.truckms.service_v2.functions_to_test import complex_func2, func_return_val, func_return_dict
import os.path as osp
import traceback
import pytest


class DummyObject:
    pass


@pytest.mark.xfail(reason="complex_func2 has Callables as arguments, which for security reasons will not be implemented now")
def test_create_p2p_clientworker_app(tmpdir):
    file_path = osp.join(tmpdir, "file.txt")
    with open(file_path, "w") as f:
        f.write("data")

    db_url = osp.join(tmpdir, "dburl")
    db = "p2p"
    col = complex_func2.__name__

    db_url_remote = osp.join(tmpdir, "dburl_remote")
    db_url_client_worker = osp.join(tmpdir, "db_url_client_worker")
    app = create_p2p_client_app()
    decorated_func = app.register_p2p_func(db_url, can_do_locally_func=lambda: False)(complex_func2)

    broker_app = create_p2p_brokerworker_app()
    broker_app.register_p2p_func(db_url_remote, can_do_locally_func=lambda: False)(complex_func2)

    def post_func(url, **kwargs):
        data = dict()
        if 'files' in kwargs:
            data.update(kwargs["files"])
        if 'data' in kwargs:
            data.update(kwargs["data"])
        print(url)
        test_broker = broker_app.test_client()
        return test_broker.post('/'.join(url.split("/")[3:]), data=data)

    from truckms.service_v2.brokerworker import p2p_brokerworker
    from truckms.service_v2.userclient import p2p_client
    from truckms.service_v2.clientworker import clientworker
    from truckms.service_v2 import p2pdata
    mocked_requests = DummyObject()
    mocked_requests.post = post_func
    p2p_brokerworker.requests = mocked_requests
    p2pdata.requests = mocked_requests
    p2p_client.self_is_reachable = lambda: None
    p2p_client.select_lru_worker = lambda port: ("ip", "port")
    clientworker.get_available_brokers = lambda local_port: [{"ip": "ip", "port": "port"}]
    clientworker.requests = mocked_requests

    try:
        data = open(file_path, "rb")
        decorated_func(identifier="x", int_arg=10, str_arg="str", file_arg=data, func_arg=func_return_val,
                       func_arg_ret_dict=func_return_dict)
        assert True
        broker_app.worker_pool.close()
        broker_app.worker_pool.join()


        # client worker enters the action
        clientworker_app = create_p2p_clientworker_app()
        do_work_func = clientworker_app.register_p2p_func(db_url_client_worker, db, col)(complex_func2)
        do_work_func()

        item = find(db_url_remote, db, col, {})[0]
        assert item['identifier'] == 'x'
        assert item['int_arg'] == 10
        assert item['str_arg'] == 'str'
        assert item['nodes'] == []
        assert item['current_address'] is None
        assert item['file_arg'] == osp.join(tmpdir, db_url_remote, db, col, 'file.txt')
        assert item['func_arg_ret_dict_key'] == 10
        assert item['val'] == 'x,10,str,10'

        returned_item = decorated_func(identifier="x", int_arg=10, str_arg="str", file_arg=data,
                                       func_arg=func_return_val, func_arg_ret_dict=func_return_dict)
        assert returned_item['val'] == 'x,10,str,10'
        assert returned_item['results_file'] == osp.join(tmpdir, db_url, db, 'file.csv')
        pass
    except:
        traceback.print_exc()
        assert False
    finally:
        app.background_thread.shutdown()
