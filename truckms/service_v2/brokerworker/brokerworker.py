from flask import request, make_response
from functools import wraps, partial
from truckms.service_v2.api import P2PFlaskApp, P2PBlueprint
from truckms.service.worker.server import create_worker_p2pblueprint
import tinymongo
import time
import multiprocessing
from truckms.service.worker.server import analyze_movie
from truckms.service_v2.p2pdata import create_p2p_blueprint
from truckms.service_v2.api import self_is_reachable


class P2PWorkerBlueprint(P2PBlueprint):

    def __init__(self, *args, num_workers, **kwargs):
        super(P2PWorkerBlueprint, self).__init__(*args, **kwargs)
        self.worker_pool = multiprocessing.Pool(num_workers)
        self.worker_pool.futures_list = []


def heartbeat(db_url):
    """
    Pottential vulnerability from flooding here
    """
    collection = tinymongo.TinyMongoClient(db_url)["tms"]["broker_heartbeats"]
    collection.insert_one({"time_of_heartbeat": time.time()})
    return make_response("Thank god you are alive", 200)


def create_brokerworker_blueprint(db_url, num_workers, function_registry):
    broker_bp = P2PWorkerBlueprint("brokerworker_bp", __name__, role="brokerworker", num_workers=num_workers)
    if num_workers == 0:
        broker_bp.worker_pool._processes = 0
    heartbeat_func = (wraps(heartbeat)(partial(heartbeat, db_url)))
    broker_bp.route("/heartbeat", methods=['POST'])(heartbeat_func)

    execute_function_route_func = (wraps(execute_function)(partial(execute_function, broker_bp.worker_pool, function_registry)))
    broker_bp.route("/execute_function/<func_name>/<resource>", methods=['GET'])(execute_function_route_func)

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


def worker_heartbeats(db_url):
    """
    If a client worker (worker that cannot be reachable and asks a broker for work) is available, then it will send a
    signal to the broker and ask for work.
    """
    collection = tinymongo.TinyMongoClient(db_url)["tms"]["broker_heartbeats"]
    result = collection.find({})
    boolean = False
    for r in result:
        boolean = ((time.time() - r["time_of_heartbeat"]) / 60) < 20
        if boolean:
            break
    return boolean


def execute_function(worker_pool, function_registry, func_name, resource):
    """
    Overwritten route from truckms.service.worker.server
    """
    analysis_func = function_registry[func_name]
    # TODO check compatibility between the requested function and the requested resource
    #  also make the monitoring possible. like the user should see that something crashed in worker pool

    # res = worker_pool.apply_async(func=analyze_and_updatedb, args=(db_url, filepath, analysis_func))
    if pool_can_do_more_work(worker_pool): # and worker_heartbeats(db_url)
        res = worker_pool.apply_async(func=analysis_func, args=(resource,))
        worker_pool.futures_list.append(res)
    else:
        # will wait for a worker client to analyze the data
        pass
    return make_response("Check later for the results", 200)


def create_broker_microservice(up_dir, db_url, num_workers=0) -> P2PFlaskApp:
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

    function_registry = {"analyze_movie", partial(analyze_movie, max_operating_res=320, skip=0)}
    broker_bp = create_brokerworker_blueprint(db_url, num_workers, function_registry=function_registry)

    p2pdata_bp = create_p2p_blueprint(up_dir, db_url, current_address_func=self_is_reachable)

    app.register_blueprint(broker_bp)
    app.register_blueprint(p2pdata_bp)
    return app
