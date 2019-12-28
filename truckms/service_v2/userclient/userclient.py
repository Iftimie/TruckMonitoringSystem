from truckms.service.gui_interface import open_browser_func, image2htmlstr, html_imgs_generator, ignore_remote_addresses
from truckms.service.gui_interface import gui_select_file
from truckms.service_v2.api import P2PFlaskApp
from typing import Tuple
from flaskwebgui import FlaskUI
from flask_bootstrap import Bootstrap
import os.path as osp
from flask import redirect, url_for, make_response, render_template, request
from truckms.service.worker.user_client import get_available_nodes
from threading import Thread
from functools import partial
import tinymongo
from truckms.service_v2.p2pdata import p2p_pull_update_one, default_deserialize, p2p_push_update_one, p2p_insert_one
import multiprocessing
from truckms.service.worker.server import analyze_movie
from truckms.service.worker.user_client import select_lru_worker, evaluate_workload
import logging
from typing import Callable
import traceback
import time
import requests
from datetime import datetime
logger = logging.getLogger(__name__)


# TODO the callable here. use type F = TypeVar[‘F’, bound=Callable[..., Any]]
#  def transparent(func: F) -> F:
def analyze_and_updatedb(video_path: str, db_url: str, analysis_func: Callable[[
                                                                                   str,
                                                                                   Callable[[int, int], None]
                                                                               ], str]):
    """
    Args:
        db_url: url for database
        video_path: path to a file on the local disk
        analysis_func: a function that receives an argument with the video path and returns the path to results.csv
            the function OPTIONALLY (can be None) receives a progress_hook

    Return:
        path to results file
    """
    destination = None
    try:

        filter_ = {"identifier": osp.basename(video_path)}
        logger.info("Started processing file")

        def progress_hook(current_index, end_index):
            update_ = {"progress": current_index / end_index * 100.0}
            p2p_push_update_one(db_url, "tms", "movie_statuses", filter_, update_)
            time.sleep(0.5)
            logger.info("Done % {}".format(update_['progress']))

        # to do refactor this. this should not look like this
        destination = analysis_func(video_path, progress_hook=progress_hook)
        update_ = {"progress": 100.0, "results": open(destination, 'rb')}
        p2p_push_update_one(db_url, "tms", "movie_statuses", filter_, update_)
        logger.info("Finished processing file")
    except:
        logger.info(traceback.format_exc())
    return destination


def get_job_dispathcher(db_url, num_workers, max_operating_res, skip, local_port, analysis_func=None):
    """
    Creates a function that is able to dispatch work. Work can be done locally or remote.

    Args:
        db_url: url for database. used to store information about the received video files
        num_workers: how many concurrent jobs should be done locally before dispatching to a remote worker
        max_operating_res: operating resolution. bigger resolution will yield better detections
        skip: how many frames should be skipped when processing, recommended 0
        local_port: port for making requests to the bookeeper service in order to find the available workers
        analysis_func: OPTIONAL.
    Return:
        function that can be called with a video_path
    """
    worker_pool = multiprocessing.Pool(num_workers)
    list_futures = []  # todo this list_futures should be removed. future responses should stay in database if are needed

    if analysis_func is None:
        analysis_func = partial(analyze_movie, max_operating_res=max_operating_res, skip=skip)

    def dispatch_work(video_path):
        lru_ip, lru_port = select_lru_worker(local_port)
        data = {"identifier": osp.basename(video_path), "video_path": open(video_path, 'rb'), "results": None,
                "time_of_request": time.time(), "progress": 0.0}
        #delete this
        def evaluate_workload():
            return False
        if evaluate_workload() or lru_ip is None:
            # do not remove this. this is useful. we don't want to upload in broker (waste time and storage when we want to process locally
            nodes = []
            p2p_insert_one(db_url, "tms", "movie_statuses", data, nodes)
            res = worker_pool.apply_async(func=analyze_and_updatedb, args=(db_url, video_path, analysis_func))
            list_futures.append(res)
            logger.info("Analyzing file locally")
        else:
            nodes = [str(lru_ip)+":"+str(lru_port)]
            p2p_insert_one(db_url, "tms", "movie_statuses", data, nodes)
            requests.post("http://{}:{}/execute_function/analyze_and_updatedb/{}".format(lru_ip, lru_port, data["identifier"]))
            # TODO call remote func with identifier
            logger.info("Dispacthed work to {},{}".format(lru_ip, lru_port))

    return dispatch_work, worker_pool, list_futures


def create_guiservice(db_url: str, dispatch_work_func: callable, port: int) -> Tuple[FlaskUI, P2PFlaskApp]:
    """
    This factory method creates the FlaskUI and P2PFlaskApp. In order to start the application, the method run of
    FlaskUI object must be called. The P2PFlaskApp object can be used to add additional blueprints.

    Args:
        db_url: url to a database
        dispatch_work_func: function that receives a string as argument. This argument is a video file path. The function
            should do something with that file
        port: port for the GUI service

    Returns:
        FlaskUI object
        P2PFlaskApp object
    """

    app = P2PFlaskApp(__name__, template_folder=osp.join(osp.dirname(__file__), '..', '..', 'service', 'templates'),
                      static_folder=osp.join(osp.dirname(__file__), '..', '..', 'service', 'templates', 'assets'))

    Bootstrap(app)

    @app.route('/file_select')
    @ignore_remote_addresses
    def file_select():
        fname = gui_select_file()
        if fname != '':
            dispatch_work_func(fname)
            return redirect(url_for("check_status"))
        else:
            return redirect(url_for("index"))

    @app.route('/')
    @ignore_remote_addresses
    def root():
        return redirect(url_for("index"))

    @app.route('/index')
    @ignore_remote_addresses
    def index():
        resp = make_response(render_template("index.html"))
        return resp

    @app.route("/show_nodestates")
    @ignore_remote_addresses
    def show_workers():
        node_list = get_available_nodes(local_port=port)
        resp = make_response(render_template("show_nodestates.html", worker_list=node_list))
        return resp

    @app.route('/check_status')
    @ignore_remote_addresses
    def check_status():
        items = list(tinymongo.TinyMongoClient(db_url)["tms"]["movie_statuses"].find({}))

        for item in items:
            up_dir = osp.dirname(item["video_path"])
            if item["results"] is None:
                p2p_pull_update_one(db_url, "tms", "movie_statuses", {"identifier": item['identifier']},
                                    req_keys=["results", "progress"], hint_file_keys=["results"], deserializer=partial(default_deserialize, up_dir=up_dir))

        # VideoStatuses.remove_dead_requests(session)
        # TODO call remove_dead_requests or insteand of removing, just restart them
        items = list(tinymongo.TinyMongoClient(db_url)["tms"]["movie_statuses"].find({}))
        video_items = []

        # TODO also render on HTML the time of execution if it exists
        # TODO why the hell I am putting the query results in a list???? I should pass directly the query
        for item in items:
            video_items.append({'identifier': item['identifier'],
                                'status': 'ready' if item['results'] is not None else 'processing',
                                'progress': item['progress'],
                                'time_of_request': datetime.utcfromtimestamp(item['time_of_request']).strftime(
                                    "%m/%d/%Y, %H:%M:%S")})
        partial_destination_url = '/show_video?identifier='
        resp = make_response(render_template("check_status.html", partial_destination_url=partial_destination_url,
                                             video_items=video_items))
        return resp

    @app.route('/show_video')
    @ignore_remote_addresses
    def show_video():
        filter_arg = request.args.get('identifier')
        item = list(tinymongo.TinyMongoClient(db_url)["tms"]["movie_statuses"].find({'identifier': filter_arg}))[0]

        plots_gen = html_imgs_generator(item['video_path'], item['results'])

        try:
            first_image = next(plots_gen)
            resp = make_response(render_template("show_video.html", first_image_str=first_image, images=plots_gen))
        except StopIteration:
            resp = make_response(render_template("show_video.html", first_image_str='', images=[]))
        return resp

    uiapp = FlaskUI(app, host="0.0.0.0", port=port)
    uiapp.browser_thread = Thread(
        target=partial(open_browser_func, uiapp, localhost='http://127.0.0.1:{}/'.format(port)))

    return uiapp, app
