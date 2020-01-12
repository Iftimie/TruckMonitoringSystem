from mock import Mock
from truckms.service_v2.p2pdata import p2p_pull_update_one, default_deserialize
from functools import partial
from pprint import pprint
class DummyObject:
    pass

def test_guiservice(tmpdir):
    from truckms.service_v2.userclient.userclient import create_guiservice, get_job_dispathcher
    from truckms.service_v2.userclient import userclient
    from truckms.service_v2.brokerworker.brokerworker import create_brokerworker_microservice
    from truckms.service.bookkeeper import create_bookkeeper_p2pblueprint
    import os
    from truckms.service_v2.userclient import userclient
    from truckms.service_v2 import p2pdata
    import tinymongo
    resource_file = os.path.join(os.path.dirname(__file__), '..', 'service', 'data', 'cut.mkv')

    port = 5000

    # create here the brokerworker
    db_url_broker_app = os.path.join(tmpdir, 'brokerworker.db')
    broker_app = create_brokerworker_microservice(tmpdir, db_url_broker_app, num_workers=1)
    broker_worker_pool = broker_app._blueprints["brokerworker_bp"].worker_pool

    bookkeeper_bp = create_bookkeeper_p2pblueprint(local_port=port, app_roles=broker_app.roles,
                                                   discovery_ips_file="discovery_ips_client.txt")
    broker_app.register_blueprint(bookkeeper_bp)
    broker_app_test_client = broker_app.test_client()


    userclient.gui_select_file = Mock(return_value=resource_file)
    userclient.select_lru_worker = Mock(return_value=("dummy_ip", "dummy_port"))
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
    userclient.requests = mocked_requests
    p2pdata.requests = mocked_requests

    db_url = os.path.join(tmpdir, 'guiinterface.db')

    dispatch_work, _, _ = get_job_dispathcher(db_url, 1, 5000)
    uiapp, app = create_guiservice(db_url, dispatch_work_func=dispatch_work, port=port)

    bookkeeper_bp = create_bookkeeper_p2pblueprint(local_port=port, app_roles=app.roles, discovery_ips_file="discovery_ips_client.txt")
    app.register_blueprint(bookkeeper_bp)

    client = app.test_client()
    res = client.get('/file_select')

    broker_worker_pool.close()
    broker_worker_pool.join()

    item_on_broker = list(tinymongo.TinyMongoClient(db_url_broker_app)["tms"]["movie_statuses"].find())[0]
    assert item_on_broker['results'] is not None

    item_on_user = list(tinymongo.TinyMongoClient(db_url)["tms"]["movie_statuses"].find())[0]
    assert item_on_user['results'] is None
    p2p_pull_update_one(db_url, "tms", "movie_statuses", {"identifier": item_on_user['identifier']},
                        req_keys=["results", "progress"], hint_file_keys=["results"],
                        deserializer=partial(default_deserialize, up_dir=tmpdir))
    item_on_user = list(tinymongo.TinyMongoClient(db_url)["tms"]["movie_statuses"].find())[0]
    assert item_on_user['results'] is not None
    pprint(item_on_user)
    pprint(item_on_broker)
