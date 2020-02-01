from truckms.service_v2.brokerworker.p2p_brokerworker import create_p2p_brokerworker_app, call_remote_func
from tests.truckms.service_v2.functions_to_test import func_return_dict, func_return_val, complex_func
from truckms.service_v2.p2pdata import p2p_insert_one, get_key_interpreter_by_signature, find
from truckms.service_v2.userclient.p2p_client import decorate_update_callables
import os.path as osp


def test_create_p2p_brokerworker_app(tmpdir):
    file_path = osp.join(tmpdir, "file.txt")
    with open(file_path, "w") as f:
        f.write("data")

    data = open(file_path, "r")
    data.close()

    app = create_p2p_brokerworker_app(tmpdir)
    db_url, db, col = osp.join(tmpdir, "dburl"), "a", "b"
    app.register_p2p_func(db_url, db, col)(complex_func)

    identifier = 'x'
    kwargs = {'identifier': identifier, 'int_arg':10, 'str_arg':"str", 'file_arg':data, 'func_arg':func_return_val, 'func_arg_ret_dict':func_return_dict}
    kwargs = decorate_update_callables(db_url, db, col, kwargs)
    ki = get_key_interpreter_by_signature(complex_func)


    p2p_insert_one(db_url, db, col, kwargs, [], key_interpreter=ki)
    test_client = app.test_client()
    res=test_client.get('/execute_function/{db}/{col}/{fname}/{identifier}'.format(db=db, col=col, fname=complex_func.__name__,
                                                                                   identifier=identifier))
    app.worker_pool.close()
    app.worker_pool.join()
    res = find(db_url, db, col, {}, ki)
    print(res)