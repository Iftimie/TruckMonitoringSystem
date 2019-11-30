import multiprocessing
from functools import partial
from truckms.service.worker.server import analyze_movie, analyze_and_updatedb
import requests
from truckms.service.model import create_session, VideoStatuses
import os
import GPUtil
from truckms.service.bookkeeper import NodeState
import time
import logging
import traceback
logger = logging.getLogger(__name__)


def evaluate_workload():
    """
    Returns a number between 0 and 1 that signifies the workload on the current PC. 0 represents no workload,
    1 means no more work can be done efficiently.
    """
    try:
        deviceID = GPUtil.getFirstAvailable(order='first', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900,
                                            verbose=False)
        if len(deviceID) != 0:
            return True
        else:
            return False
    except RuntimeError:
        return False


def select_lru_worker(local_port):
    """
    Selects the least recently used worker from the known states and returns its IP and PORT
    """
    try:
        res = requests.get('http://localhost:{}/node_states'.format(local_port)).json()  # will get the data defined above
    except:
        logger.info(traceback.format_exc())
        return None, None

    res1 = [item for item in res if 'worker' in item['node_type'] or 'broker' in item['node_type']]
    if len(res1) == 0:
        logger.info("No worker or broker available")
        return None, None
    res1 = sorted(res1, key=lambda x: x['workload'])
    while res1:
        try:
            response = requests.get('http://{}:{}/echo'.format(res1[0]['ip'], res1[0]['port']))
            if response.status_code != 200:
                res1.pop(0)
        except:
            res1.pop(0)
            logger.info(traceback.format_exc())

    if len(res1) == 0:
        logger.info("No worker or broker available")
        return None, None
    return res1[0]['ip'], res1[0]['port']


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
        #delete this
        # def evaluate_workload():
        #     return False
        if evaluate_workload() or lru_ip is None:
            # do not remove this. this is useful. we don't want to upload in broker (waste time and storage when we want to process locally
            res = worker_pool.apply_async(func=analyze_and_updatedb, args=(db_url, video_path, analysis_func))
            list_futures.append(res)
            logger.info("Analyzing file locally")
        else:
            session = create_session(db_url)
            VideoStatuses.add_video_status(session, file_path=video_path, results_path=None, remote_ip=lru_ip, remote_port=lru_port)
            data = {"max_operating_res": max_operating_res, "skip": skip}
            files = {os.path.basename(video_path): open(video_path, 'rb')}
            res = requests.post('http://{}:{}/upload_recordings'.format(lru_ip, lru_port), data=data, files=files)
            assert res.content == b"Files uploaded and started runniing the detector. Check later for the results"
            session.close()
            time.sleep(2)
            logger.info("Dispacthed work to {},{}".format(lru_ip, lru_port))
            # do work remotely

    return dispatch_work, worker_pool, list_futures
