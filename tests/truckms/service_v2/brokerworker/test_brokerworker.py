from truckms.service_v2.brokerworker.brokerworker import worker_heartbeats, execute_function
import os.path as osp
import tinymongo
import time
import multiprocessing


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
    execute_function(db_url, worker_pool, function_registry, "f", None)
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
    execute_function(db_url, worker_pool, function_registry, "f", None)
    worker_pool.close()
    worker_pool.join()
    assert len(worker_pool.futures_list) == 0


def sync_func(event):
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

    function_registry = {"sync_func": sync_func,
                         "f": f}
    m = multiprocessing.Manager()
    sync_event = m.Event()
    execute_function(db_url, worker_pool, function_registry, "sync_func", sync_event)
    execute_function(db_url, worker_pool, function_registry, "f", None)
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

    function_registry = {"sync_func": sync_func,
                         "f": f}
    m = multiprocessing.Manager()
    sync_event = m.Event()
    execute_function(db_url, worker_pool, function_registry, "sync_func", sync_event)
    execute_function(db_url, worker_pool, function_registry, "f", None)
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

    function_registry = {"sync_func": sync_func,
                         "f": f}
    m = multiprocessing.Manager()
    sync_event = m.Event()
    collection = tinymongo.TinyMongoClient(db_url)["tms"]["broker_heartbeats"]
    collection.insert_one({"time_of_heartbeat": time.time()})

    execute_function(db_url, worker_pool, function_registry, "sync_func", sync_event)
    execute_function(db_url, worker_pool, function_registry, "f", None)
    sync_event.set()
    worker_pool.close()
    worker_pool.join()
    assert len(worker_pool.futures_list) == 1
