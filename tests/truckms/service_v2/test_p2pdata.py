from truckms.service_v2.p2pdata import update_one, p2p_insert_one, create_p2p_blueprint, p2p_update_one
from truckms.service_v2.api import P2PFlaskApp, self_is_reachable
import os
from copy import deepcopy
from pprint import pprint


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

    def post_func(url, **kwargs):
        data = dict()
        data.update(kwargs["files"])
        data.update(kwargs["json"])
        client = urls[url]
        client.post('/'.join(url.split("/")[3:]), data=data)

    i = 0
    def create_app():
        nonlocal i
        remote_db_url = os.path.join(tmpdir, "remote_mongodb_{}".format(i))
        i += 1
        app = P2PFlaskApp(__name__)
        p2p_bp = create_p2p_blueprint(up_dir=tmpdir, db_url=remote_db_url, post_func=post_func)
        app.register_blueprint(p2p_bp)
        return app

    nodes = ["localhost:0000"]
    urls = {"http://{}/insert_one/{}/{}".format(node, db, col): create_app().test_client() for node in nodes}


    p2p_insert_one(db_url, db, col, data, nodes, post_func=post_func, current_address_func=lambda :self_is_reachable("0000"))
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

    def post_func(url, **kwargs):
        data = dict()
        data.update(kwargs["files"])
        data.update(kwargs["json"])
        client = urls[url]
        client.post('/'.join(url.split("/")[3:]), data=data)

    i = 0
    def create_app():
        nonlocal i
        remote_db_url = os.path.join(tmpdir, "remote_mongodb_{}".format(i))
        i += 1
        app = P2PFlaskApp(__name__)
        p2p_bp = create_p2p_blueprint(tmpdir, remote_db_url, post_func=post_func)
        app.register_blueprint(p2p_bp)
        return app

    nodes = ["localhost:0000"]
    urls = {"http://{}/insert_one/{}/{}".format(node, db, col): create_app().test_client() for node in nodes}


    p2p_insert_one(db_url, db, col, data, nodes, post_func=post_func, current_address_func=lambda : self_is_reachable("0000"))
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


def test_p2p_update_one(tmpdir):
    import tinymongo

    db_url = os.path.join(tmpdir, "mongodb")
    db, col = "mydb", "movie_statuses"
    data = {"res": 320, "name": "somename", "key":"value0"}

    def post_func(url, **kwargs):
        data = dict()
        data.update(kwargs["files"])
        data.update(kwargs["json"])
        client = urls[url]
        client.post('/'.join(url.split("/")[3:]), data=data)

    i = 0
    def create_app():
        nonlocal i
        remote_db_url = os.path.join(tmpdir, "remote_mongodb_{}".format(i))
        i += 1
        app = P2PFlaskApp(__name__)
        p2p_bp = create_p2p_blueprint(up_dir=tmpdir, db_url=remote_db_url, post_func=post_func)
        app.register_blueprint(p2p_bp)
        return app

    nodes = ["localhost:0000"]
    client_ = create_app().test_client()
    urls = {"http://{}/insert_one/{}/{}".format(nodes[0], db, col): client_,
            "http://{}/update_one/{}/{}".format(nodes[0], db, col): client_}


    p2p_insert_one(db_url, db, col, data, nodes, post_func=post_func, current_address_func=lambda : self_is_reachable("0000"))
    new_data = deepcopy(data)
    new_data["name"] = "othername"
    p2p_insert_one(db_url, db, col, new_data, nodes, post_func=post_func, current_address_func=lambda : self_is_reachable("0000"))
    collection = tinymongo.TinyMongoClient(db_url)[db][col]
    collection = list(collection.find())
    assert len(collection) == 2
    pprint(collection)
    print()

    filter = {"res": 320, "name": "othername"}
    update_value = {"key": "value1"}
    p2p_update_one(db_url, db, col, filter, update_value, post_func=post_func)
    collection = tinymongo.TinyMongoClient(db_url)[db][col]
    collection = list(collection.find())
    assert len(collection) == 2
    pprint(collection)

    collection_remote = tinymongo.TinyMongoClient(os.path.join(tmpdir, "remote_mongodb_{}".format(0)))[db][col]
    collection_remote = list(collection_remote.find())
    assert len(collection_remote) == 2
    print(collection_remote)
    for i in range(len(collection_remote)):
        for k in collection_remote[i]:
            if k in ["_id", "nodes"]: continue
            if "source_code_file" == k:
                assert os.path.basename(collection_remote[0][k]) == os.path.basename(collection[0][k])
            else:
                assert collection_remote[0][k] == collection[0][k]


def test_p2p_update_one_with_files(tmpdir):
    import tinymongo

    db_url = os.path.join(tmpdir, "mongodb")
    db, col = "mydb", "movie_statuses"
    data = {"res": 320, "name": "somename", "key":"value0", "source_code_file": None}

    def post_func(url, **kwargs):
        data = dict()
        data.update(kwargs["files"])
        data.update(kwargs["json"])
        client = urls[url]
        client.post('/'.join(url.split("/")[3:]), data=data)

    i = 0
    def create_app():
        nonlocal i
        remote_db_url = os.path.join(tmpdir, "remote_mongodb_{}".format(i))
        i += 1
        app = P2PFlaskApp(__name__)
        p2p_bp = create_p2p_blueprint(up_dir=tmpdir, db_url=remote_db_url, post_func=post_func)
        app.register_blueprint(p2p_bp)
        return app

    nodes = ["localhost:0000"]
    client_ = create_app().test_client()
    urls = {"http://{}/insert_one/{}/{}".format(nodes[0], db, col): client_,
            "http://{}/update_one/{}/{}".format(nodes[0], db, col): client_}


    p2p_insert_one(db_url, db, col, data, nodes, post_func=post_func)
    new_data = deepcopy(data)
    new_data["name"] = "othername"
    p2p_insert_one(db_url, db, col, new_data, nodes, post_func=post_func, current_address_func=lambda : self_is_reachable("0000"))
    collection = tinymongo.TinyMongoClient(db_url)[db][col]
    collection = list(collection.find())
    assert len(collection) == 2
    pprint(collection)
    print()

    filter = {"res": 320, "name": "othername"}
    update_value = {"key": "value1",
                    "source_code_file": open(__file__, 'rb')}
    p2p_update_one(db_url, db, col, filter, update_value, post_func=post_func)
    collection = tinymongo.TinyMongoClient(db_url)[db][col]
    collection = list(collection.find())
    assert len(collection) == 2
    pprint(collection)

    collection_remote = tinymongo.TinyMongoClient(os.path.join(tmpdir, "remote_mongodb_{}".format(0)))[db][col]
    collection_remote = list(collection_remote.find())
    assert len(collection_remote) == 2
    pprint(collection_remote)
    for i in range(len(collection_remote)):
        for k in collection_remote[i]:
            if k in ["_id", "nodes"]: continue
            if "source_code_file" == k:
                if collection_remote[0][k] == None:
                    assert collection_remote[0][k] == collection[0][k]
                else:
                    assert os.path.basename(collection_remote[0][k]) == os.path.basename(collection[0][k])
            else:
                assert collection_remote[0][k] == collection[0][k]


def test_p2p_link(tmpdir):
    import tinymongo

    db, col = "mydb", "movie_statuses"
    node1data = {"res": 320, "name": "somename", "key": "value0"}
    nodes = ["0000:0000", "1111:1111", "2222:2222"]
    db_url0 = os.path.join(tmpdir, "mongodb0")
    db_url1 = os.path.join(tmpdir, "mongodb1")
    db_url2 = os.path.join(tmpdir, "mongodb2")

    def post_func(url, **kwargs):
        data = dict()
        data.update(kwargs["files"])
        data.update(kwargs["json"])
        client = urls[url]
        return client.post('/'.join(url.split("/")[3:]), data=data)

    def create_app(db_url, current_addr_func):
        app = P2PFlaskApp(__name__)
        p2p_bp = create_p2p_blueprint(up_dir=tmpdir, db_url=db_url, post_func=post_func, current_address_func=current_addr_func)
        app.register_blueprint(p2p_bp)
        return app

    client0 = create_app(db_url0, lambda :"0000:0000").test_client()
    client1 = create_app(db_url1, lambda :"1111:1111").test_client()
    client2 = create_app(db_url2, lambda :"2222:2222").test_client()
    urls = {"http://{}/insert_one/{}/{}".format(nodes[0], db, col): client0,
            "http://{}/update_one/{}/{}".format(nodes[0], db, col): client0,
            "http://{}/insert_one/{}/{}".format(nodes[1], db, col): client1,
            "http://{}/update_one/{}/{}".format(nodes[1], db, col): client1,
            "http://{}/insert_one/{}/{}".format(nodes[2], db, col): client2,
            "http://{}/update_one/{}/{}".format(nodes[2], db, col): client2
            }

    # this will mimic the following connection:
    # node 0 can reach node 1, and can be reached by 1
    # node 1 can be reached by any node, and can reach node 0
    # node 2 can reach node 1, but cannot be reached by any node

    # we insert here and remotely
    p2p_insert_one(db_url0, db, col, node1data, nodes=["1111:1111"], post_func=post_func, current_address_func=lambda:"0000:0000")
    # we update remotely the node list. assume node 1 can reach node 0
    node1filter_ = {"name": "somename"}
    node1update_data = list(tinymongo.TinyMongoClient(db_url1)[db][col].find(node1filter_))[0]
    node1update_data = {k: node1update_data[k] for k in node1update_data if k != "_id"}
    # node1update_data["nodes"].extend(["0000:0000"])
    update_one(db_url1, db, col, node1filter_, node1update_data)
    # the client worker will download the data and will insert it by itself
    node2update_data = deepcopy(node1update_data)
    node2filter_ = node1filter_
    node2update_data["nodes"] = ["1111:1111"]
    node2update_data["current_address"] = "2222:2222"
    update_one(db_url2, db, col, node2filter_, node2update_data, upsert=True)


    collection0 = tinymongo.TinyMongoClient(db_url0)[db][col]
    collection1 = tinymongo.TinyMongoClient(db_url1)[db][col]
    collection2 = tinymongo.TinyMongoClient(db_url2)[db][col]
    pprint(list(collection0.find()))
    pprint(list(collection1.find()))
    pprint(list(collection2.find()))

    node2update_data = {"res": 800}
    visited_nodes = p2p_update_one(db_url2, db, col, node2filter_, node2update_data, post_func=post_func)
    assert set(visited_nodes) == set(nodes)
    collection0 = tinymongo.TinyMongoClient(db_url0)[db][col]
    collection1 = tinymongo.TinyMongoClient(db_url1)[db][col]
    collection2 = tinymongo.TinyMongoClient(db_url2)[db][col]
    assert list(collection0.find())[0]['res'] == 800
    assert list(collection1.find())[0]['res'] == 800
    assert list(collection2.find())[0]['res'] == 800
