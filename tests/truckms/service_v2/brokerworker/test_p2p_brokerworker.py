from truckms.service_v2.brokerworker.p2p_brokerworker import create_p2p_brokerworker_app
from tests.truckms.service_v2.functions_to_test import func_return_dict, func_return_val, complex_func
import os.path as osp


def test_create_p2p_brokerworker_app(tmpdir):
    app = create_p2p_brokerworker_app(tmpdir)
    db_url, db, col = osp.join(tmpdir, "dburl"), "a", "b"
    app.register_p2p_func(db_url, db, col)(complex_func)

    identifier = 'x'
    test_client = app.test_client()
    res=test_client.get('/execute_function/a/b/complex_func/x')
    print(res)