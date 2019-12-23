from truckms.service_v2.p2pdata import update_one, p2p_insert_one, create_p2p_blueprint
from truckms.service_v2.api import P2PFlaskApp
import os


def test_insert_one(tmpdir):
    import tinymongo
    db_url = os.path.join(tmpdir, "mongodb")
    db, col = "mydb", "movie_statuses"
    data = {"res": 320, "name": "somename"}

    update_one(db_url, db, col, data, data, upsert=True)
    collection = tinymongo.TinyMongoClient(db_url)[db][col]

    assert len(list(collection.find())) == 1


def test_insert_one_file(tmpdir):
    import tinymongo
    db_url = os.path.join(tmpdir, "mongodb")
    db, col = "mydb", "movie_statuses"
    data = {"res": 320, "name": "somename", "source_code": open(__file__, 'r')}

    update_one(db_url, db, col, data, data, upsert=True)
    collection = tinymongo.TinyMongoClient(db_url)[db][col]

    assert len(list(collection.find())) == 1


def test_insert_one_file_binary(tmpdir):
    import tinymongo
    db_url = os.path.join(tmpdir, "mongodb")
    db, col = "mydb", "movie_statuses"
    data = {"res": 320, "name": "somename", "source_code": open(__file__, 'rb')}

    update_one(db_url, db, col, data, data, upsert=True)
    collection = tinymongo.TinyMongoClient(db_url)[db][col]

    assert len(list(collection.find())) == 1


def test_p2p_insert_one(tmpdir):
    import tinymongo

    db_url = os.path.join(tmpdir, "mongodb")
    db, col = "mydb", "movie_statuses"
    data = {"res": 320, "name": "somename"}

    i = 0

    def create_app():
        nonlocal i
        remote_db_url = os.path.join(tmpdir, "remote_mongodb_{}".format(i))
        i += 1
        app = P2PFlaskApp(__name__)
        p2p_bp = create_p2p_blueprint(up_dir=tmpdir, db_url=remote_db_url)
        app.register_blueprint(p2p_bp)
        return app

    nodes = ["localhost:0000"]
    urls = {"http://{}/insert_one/{}/{}".format(node, db, col): create_app().test_client() for node in nodes}

    def post_func(url, **kwargs):
        data = dict()
        data.update(kwargs["files"])
        data.update(kwargs["json"])
        client = urls[url]
        client.post('/'.join(url.split("/")[3:]), data=data)

    p2p_insert_one(db_url, db, col, data, nodes, local_port="0000", post_func=post_func)
    collection = tinymongo.TinyMongoClient(db_url)[db][col]
    collection = list(collection.find())
    assert len(collection) == 1
    print(collection)

    collection_remote = tinymongo.TinyMongoClient(os.path.join(tmpdir, "remote_mongodb_{}".format(0)))[db][col]
    collection_remote = list(collection_remote.find())
    assert len(collection_remote) == 1
    print(collection_remote)
    for k in collection_remote[0]:
        if k in ["_id", "nodes"]: continue
        assert collection_remote[0][k] == collection[0][k]


def test_p2p_insert_one_with_files(tmpdir):
    import tinymongo

    db_url = os.path.join(tmpdir, "mongodb")
    db, col = "mydb", "movie_statuses"
    data = {"res": 320, "name": "somename", "source_code_file": open(__file__, 'rb')}

    i = 0
    def create_app():
        nonlocal i
        remote_db_url = os.path.join(tmpdir, "remote_mongodb_{}".format(i))
        i += 1
        app = P2PFlaskApp(__name__)
        p2p_bp = create_p2p_blueprint(tmpdir, remote_db_url)
        app.register_blueprint(p2p_bp)
        return app

    nodes = ["localhost:0000"]
    urls = {"http://{}/insert_one/{}/{}".format(node, db, col): create_app().test_client() for node in nodes}

    def post_func(url, **kwargs):
        data = dict()
        data.update(kwargs["files"])
        data.update(kwargs["json"])
        client = urls[url]
        client.post('/'.join(url.split("/")[3:]), data=data)

    p2p_insert_one(db_url, db, col, data, nodes, local_port="0000", post_func=post_func)
    collection = tinymongo.TinyMongoClient(db_url)[db][col]
    collection = list(collection.find())
    assert len(collection) == 1
    print(collection)

    collection_remote = tinymongo.TinyMongoClient(os.path.join(tmpdir, "remote_mongodb_{}".format(0)))[db][col]
    collection_remote = list(collection_remote.find())
    assert len(collection_remote) == 1
    print(collection_remote)
    for k in collection_remote[0]:
        if k in ["_id", "nodes"]: continue
        if "source_code_file" == k:
            assert os.path.basename(collection_remote[0][k]) == os.path.basename(collection[0][k])
        else:
            assert collection_remote[0][k] == collection[0][k]

def test_p2p_update(tmpdir):
    db = "mydb"
    col = "movie_statuses"

