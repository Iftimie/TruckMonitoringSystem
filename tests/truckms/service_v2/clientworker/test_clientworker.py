from truckms.service_v2.clientworker.clientworker import find_response_with_work


class DummyObject(object):
    pass


def test_find_response_with_work():
    from truckms.service_v2.clientworker import clientworker
    clientworker.get_available_brokers = lambda local_port: [{"ip": "dummy_ip", "port": "dummy_port"}]
    response = DummyObject()
    response.json = {"identifier": "dummyid"}
    clientworker.requests.get = lambda url: response

    res_json, res_broker_ip, res_broker_port = find_response_with_work("local_port", "dummy_col", "dummy_func")
    assert res_json == {"identifier": "dummyid"}
    assert res_broker_ip == "dummy_ip"
    assert res_broker_port == "dummy_port"
