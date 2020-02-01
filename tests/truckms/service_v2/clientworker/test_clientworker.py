from truckms.service_v2.clientworker.clientworker import find_response_with_work, do_work
from truckms.service_v2.brokerworker.brokerworker import create_brokerworker_microservice
import os.path as osp
from functools import partial, wraps


class DummyObject(object):
    pass


def test_find_response_with_work():
    from truckms.service_v2.clientworker import clientworker
    clientworker.get_available_brokers = lambda local_port: [{"ip": "dummy_ip", "port": "dummy_port"}]
    response = DummyObject()
    response.json = {"identifier": "dummyid"}
    clientworker.requests.post = lambda url, timeout: response

    res_json, res_broker_ip, res_broker_port = find_response_with_work("local_port", "dummy_col", "dummy_func", "dummy_name")
    assert res_json == {"identifier": "dummyid"}
    assert res_broker_ip == "dummy_ip"
    assert res_broker_port == "dummy_port"


def test_do_work(tmpdir):
    from truckms.service_v2.clientworker import clientworker
    from truckms.service_v2 import p2pdata
    import tinymongo
    clientworker.find_response_with_work = lambda p, c, f: ({"identifier": "dummyid"}, "dummy_ip", "dummy_port")

    db_url_remote = osp.join(tmpdir, "dummy_remote.db")
    data_remote = {"identifier": "dummyid", "video_path": open(osp.join(osp.dirname(__file__), "..", "..", "service", "data", "cut.mkv"), 'rb'),
                   "arg_0": 0}
    p2pdata.p2p_insert_one(db_url_remote, "tms", "movie_statuses", data_remote, [])

    broker_app = create_brokerworker_microservice(tmpdir, db_url_remote, 1)
    broker_app_test_client = broker_app.test_client()

    mocked_requests = DummyObject()
    def post_func(url, *args, **kwargs):
        stripped_url = '/'.join(url.split("/")[3:])
        data = dict()
        if "files" in kwargs:
            data.update(kwargs["files"])
        if "data" in kwargs:
            data.update(kwargs["data"])
        return broker_app_test_client.post(stripped_url, data=data)

    mocked_requests.post = post_func
    p2pdata.requests = mocked_requests

    db_url_local = osp.join(tmpdir, "dummy.db")
    port = 5000
    def original_func(video_path, arg_0, arg_1, arg_2=2):
        assert video_path is not None and arg_0 is not None
        return {"results": open(osp.join(osp.dirname(__file__), "..", "..", "service", "data", "cut.csv"), 'rb')}
    original_func.hint_args_are_files = ["video_path"]
    partial_func = wraps(original_func)(partial(original_func, arg_1=1))

    do_work(tmpdir, db_url_local, port, partial_func, db="tms", collection="movie_statuses")


    collection = list(tinymongo.TinyMongoClient(db_url_remote)["tms"]["movie_statuses"].find({}))
    assert len(collection) == 1
    assert "results" in collection[0]

