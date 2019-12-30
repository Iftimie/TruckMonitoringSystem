from truckms.service_v2.brokerworker.brokerworker import worker_heartbeats, execute_function, search_work
import os.path as osp
import tinymongo
import time
import multiprocessing
from functools import partial


def test_worker_heartbeats(tmpdir):
    db_url = osp.join(tmpdir, "broker.db")
    collection = tinymongo.TinyMongoClient(db_url)["tms"]["broker_heartbeats"]
    collection.insert_one({"time_of_heartbeat": time.time()})
    collection.insert_one({"time_of_heartbeat": time.time()})
    collection.insert_one({"time_of_heartbeat": time.time()})

    assert worker_heartbeats(db_url) == True


def test_not_worker_heartbeats(tmpdir):
    db_url = osp.join(tmpdir, "broker.db")
    collection = tinymongo.TinyMongoClient(db_url)["tms"]["broker_heartbeats"]
    collection.insert_one({"time_of_heartbeat": time.time() - 60 * 21})

    assert worker_heartbeats(db_url) == False


def f(resource):
    return True


def test_execute_function(tmpdir):
    from truckms.service_v2.brokerworker import brokerworker

    brokerworker.make_response = lambda s, c: None
    db_url = osp.join(tmpdir, "broker.db")
    num_workers = 1
    worker_pool = multiprocessing.Pool(num_workers)
    worker_pool.futures_list = []

    function_registry = {"f": f}
    collection = tinymongo.TinyMongoClient(db_url)["tms"]["dummy"]
    collection.insert_one({"identifier": "mov1", "timestamp": time.time()})

    execute_function(db_url, worker_pool, function_registry, "dummy", "f", "mov1")
    worker_pool.close()
    worker_pool.join()
    assert worker_pool.futures_list[0]._value is True


def test_execute_function2(tmpdir):
    from truckms.service_v2.brokerworker import brokerworker

    brokerworker.make_response = lambda s, c: None
    db_url = osp.join(tmpdir, "broker.db")
    num_workers = 0
    worker_pool = multiprocessing.Pool(num_workers if num_workers > 0 else 1)
    worker_pool._processes = num_workers
    worker_pool.futures_list = []

    function_registry = {"f": f}
    collection = tinymongo.TinyMongoClient(db_url)["tms"]["dummy"]
    collection.insert_one({"identifier": "mov1", "timestamp": time.time()})
    execute_function(db_url, worker_pool, function_registry, "dummy", "f", "mov1")
    worker_pool.close()
    worker_pool.join()
    assert len(worker_pool.futures_list) == 0


def sync_func(resource, event):
    event.wait()
    return "ok"


def test_execute_function3(tmpdir):
    from truckms.service_v2.brokerworker import brokerworker

    brokerworker.make_response = lambda s, c: None
    db_url = osp.join(tmpdir, "broker.db")
    num_workers = 1
    worker_pool = multiprocessing.Pool(num_workers if num_workers > 0 else 1)
    worker_pool._processes = num_workers
    worker_pool.futures_list = []

    m = multiprocessing.Manager()
    sync_event = m.Event()
    function_registry = {"sync_func": partial(sync_func, event=sync_event),
                         "f": f}
    collection = tinymongo.TinyMongoClient(db_url)["tms"]["dummy"]
    collection.insert_one({"identifier": "mov1", "timestamp": time.time()})
    collection.insert_one({"identifier": "mov2", "timestamp": time.time()})

    execute_function(db_url, worker_pool, function_registry, "dummy", "sync_func", "mov1")
    execute_function(db_url, worker_pool, function_registry, "dummy", "f", "mov2")
    sync_event.set()
    worker_pool.close()
    worker_pool.join()
    assert len(worker_pool.futures_list) == 2
    assert worker_pool.futures_list[0]._value == "ok"
    assert worker_pool.futures_list[1]._value is True


def test_execute_function4(tmpdir):
    from truckms.service_v2.brokerworker import brokerworker

    brokerworker.make_response = lambda s, c: None
    db_url = osp.join(tmpdir, "broker.db")
    num_workers = 0
    worker_pool = multiprocessing.Pool(num_workers if num_workers > 0 else 1)
    worker_pool._processes = num_workers
    worker_pool.futures_list = []

    m = multiprocessing.Manager()
    sync_event = m.Event()
    collection = tinymongo.TinyMongoClient(db_url)["tms"]["dummy"]
    collection.insert_one({"identifier": "mov1", "timestamp": time.time()})
    collection.insert_one({"identifier": "mov2", "timestamp": time.time()})
    function_registry = {"sync_func": partial(sync_func, event=sync_event),
                         "f": f}
    execute_function(db_url, worker_pool, function_registry, "dummy", "sync_func", "mov1")
    execute_function(db_url, worker_pool, function_registry, "dummy", "f", "mov2")
    sync_event.set()
    worker_pool.close()
    worker_pool.join()
    assert len(worker_pool.futures_list) == 0


def test_execute_function5(tmpdir):
    from truckms.service_v2.brokerworker import brokerworker

    brokerworker.make_response = lambda s, c: None
    db_url = osp.join(tmpdir, "broker.db")
    num_workers = 1
    worker_pool = multiprocessing.Pool(num_workers if num_workers > 0 else 1)
    worker_pool._processes = num_workers
    worker_pool.futures_list = []

    m = multiprocessing.Manager()
    sync_event = m.Event()
    function_registry = {"sync_func": partial(sync_func, event=sync_event),
                         "f": f}
    collection = tinymongo.TinyMongoClient(db_url)["tms"]["broker_heartbeats"]
    collection.insert_one({"time_of_heartbeat": time.time()})

    collection = tinymongo.TinyMongoClient(db_url)["tms"]["dummy"]
    collection.insert_one({"identifier": "mov1", "timestamp": time.time()})
    collection.insert_one({"identifier": "mov2", "timestamp": time.time()})

    execute_function(db_url, worker_pool, function_registry, "dummy", "sync_func", "mov1")
    execute_function(db_url, worker_pool, function_registry, "dummy", "f", "mov2")
    sync_event.set()
    worker_pool.close()
    worker_pool.join()
    assert len(worker_pool.futures_list) == 1


def test_execute_function6(tmpdir):
    from truckms.service_v2.brokerworker import brokerworker

    brokerworker.make_response = lambda s, c: (s, c)
    db_url = osp.join(tmpdir, "broker.db")
    num_workers = 1
    worker_pool = multiprocessing.Pool(num_workers if num_workers > 0 else 1)
    worker_pool._processes = num_workers
    worker_pool.futures_list = []

    m = multiprocessing.Manager()
    sync_event = m.Event()
    function_registry = {"sync_func": partial(sync_func, event=sync_event),
                         "f": f}
    collection = tinymongo.TinyMongoClient(db_url)["tms"]["broker_heartbeats"]
    collection.insert_one({"time_of_heartbeat": time.time()})

    collection = tinymongo.TinyMongoClient(db_url)["tms"]["movie_statuses"]
    collection.insert_one({"identifier": "mov2", "timestamp": time.time()})

    execute_function(db_url, worker_pool, function_registry, "movie_statuses", "f", "mov2")
    res = execute_function(db_url, worker_pool, function_registry, "movie_statuses", "f", "mov2")
    assert res[0] == "Allready started"
    sync_event.set()
    worker_pool.close()
    worker_pool.join()
    assert len(worker_pool.futures_list) == 1


def test_search_work(tmpdir):
    from truckms.service_v2.brokerworker import brokerworker
    brokerworker.jsonify = lambda item: item
    db_url = osp.join(tmpdir, "broker.db")
    collection = tinymongo.TinyMongoClient(db_url)["tms"]["movie_statuses"]
    collection.insert_one({"identifier": "mov1", "started": "dummy_func", "timestamp": time.time()})
    collection.insert_one({"identifier": "mov2", "timestamp": time.time()+10})
    collection.insert_one({"identifier": "mov3", "started": "dummy_func2", "timestamp": time.time()})
    collection.insert_one({"identifier": "mov4", "started": "dummy_func3", "timestamp": time.time()})

    res = search_work(db_url, "movie_statuses", "dummy_func")
    assert res['identifier'] == "mov3"
