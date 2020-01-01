from flask import request, make_response, jsonify
from functools import wraps, partial
from truckms.service_v2.api import P2PFlaskApp, P2PBlueprint
from truckms.service.worker.server import create_worker_p2pblueprint
import tinymongo
import time
import multiprocessing
from truckms.service.worker.server import analyze_movie
from truckms.service_v2.userclient.userclient import analyze_and_updatedb
from truckms.service_v2.p2pdata import create_p2p_blueprint
from truckms.service_v2.api import self_is_reachable


class P2PWorkerBlueprint(P2PBlueprint):

    def __init__(self, *args, num_workers, **kwargs):
        super(P2PWorkerBlueprint, self).__init__(*args, **kwargs)
        self.worker_pool = multiprocessing.Pool(num_workers)
        self.worker_pool.futures_list = []


def heartbeat(db_url, db="tms"):
    """
    Pottential vulnerability from flooding here
    """
    collection = tinymongo.TinyMongoClient(db_url)[db]["broker_heartbeats"]
    collection.insert_one({"time_of_heartbeat": time.time()})
    return make_response("Thank god you are alive", 200)


def create_brokerworker_blueprint(db_url, num_workers, function_registry):
    broker_bp = P2PWorkerBlueprint("brokerworker_bp", __name__, role="brokerworker", num_workers=num_workers if num_workers!=0 else 1)
    if num_workers == 0:
        broker_bp.worker_pool._processes = 0
    heartbeat_func = (wraps(heartbeat)(partial(heartbeat, db_url)))
    broker_bp.route("/heartbeat", methods=['POST'])(heartbeat_func)

    execute_function_route_func = (wraps(execute_function)(partial(execute_function, db_url, broker_bp.worker_pool, function_registry)))
    broker_bp.route("/execute_function/<collection>/<func_name>/<resource>", methods=['POST'])(execute_function_route_func)

    search_work_func = (wraps(search_work)(partial(search_work, db_url)))
    broker_bp.route("/search_work/<collection>/<func_name>", methods=['GET'])(search_work_func)


    return broker_bp


def pool_can_do_more_work(worker_pool):
    """
    Checks if there are available workers in the pool.
    """
    count_opened = 0
    for apply_result in worker_pool.futures_list[:]:
        try:
            apply_result.get(1)
            # if not timeout exception, then we can safely remove the object
            worker_pool.futures_list.remove(apply_result)
        except:
            count_opened+=1
    if count_opened < worker_pool._processes:
        return True
    else:
        return False


def pool_can_do_more_work_later(worker_pool):
    return worker_pool._processes > 0


def worker_heartbeats(db_url, minutes=20):
    """
    If a client worker (worker that cannot be reachable and asks a broker for work) is available, then it will send a
    signal to the broker and ask for work.
    """
    collection = tinymongo.TinyMongoClient(db_url)["tms"]["broker_heartbeats"]
    result = collection.find({})
    boolean = False
    for r in result:
        boolean = ((time.time() - r["time_of_heartbeat"]) / 60) < minutes
        if boolean:
            break
    return boolean


def search_work(db_url, collection, func_name, db="tms"):

    col = list(tinymongo.TinyMongoClient(db_url)[db][collection].find({}))

    col = [item for item in col if "started" not in item or item["started"] != func_name]
    if col:
        col.sort(key=lambda item: item["timestamp"])  # might allready be sorted
        item = col[0]
        tinymongo.TinyMongoClient(db_url)[db][collection].update_one({"identifier": item["identifier"]},
                                                                        {"started": func_name})
        return jsonify({"identifier": item["identifier"]})
    else:
        return jsonify({})


def execute_function(db_url, worker_pool, function_registry, collection, func_name, resource, db="tms"):
    func_ = function_registry[func_name]
    # TODO check compatibility between the requested function and the requested resource
    #  also make the monitoring possible. like the user should see that something crashed in worker pool

    # TODO resources that have been added by the execute_function for them wasn't called should be deleted

    item = list(tinymongo.TinyMongoClient(db_url)[db][collection].find({"identifier": resource}))[0]
    is_done = "started" in item and item["started"] == func_name
    if is_done:
        return make_response("Allready started", 200)

    # res = worker_pool.apply_async(func=analyze_and_updatedb, args=(db_url, filepath, analysis_func))
    if pool_can_do_more_work(worker_pool) or (not worker_heartbeats(db_url) and pool_can_do_more_work_later(worker_pool)):
        tinymongo.TinyMongoClient(db_url)["tms"][collection].update_one({"identifier": resource},
                                                                        {"started": func_name})
        res = worker_pool.apply_async(func=func_, args=(resource,))
        worker_pool.futures_list.append(res)
    else:
        # will wait for a worker client to analyze the data
        pass
    return make_response("Check later for the results", 200)
    # TODO it should return something like a promise ?


def create_brokerworker_microservice(up_dir, db_url, num_workers=0, functions_list=None) -> P2PFlaskApp:
    """
    Args:
        up_dir: path to directory where to store video files and files with results
        db_url: url to database
        num_workers: number of workers for the worker_blueprint. In this case is 0 because this service is by default
        only a broker, however, a worker_blueprint can also be a broker and not have num_workers set on 0

    Return:
        P2PFlaskApp
    """
    app = P2PFlaskApp(__name__)
    if functions_list is None:
        functions_list = [wraps(analyze_and_updatedb)(partial(analyze_and_updatedb, db_url=db_url, analysis_func=analyze_movie))]

    function_registry = {func.__name__: func for func in functions_list}
    broker_bp = create_brokerworker_blueprint(db_url, num_workers, function_registry=function_registry)

    p2pdata_bp = create_p2p_blueprint(up_dir, db_url)  # TODO remove the need for this function current_address_func=self_is_reachable

    app.register_blueprint(broker_bp)
    app.register_blueprint(p2pdata_bp)
    return app
