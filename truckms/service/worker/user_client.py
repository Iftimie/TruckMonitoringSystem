import multiprocessing
from functools import partial
from truckms.service.worker.server import analyze_movie, analyze_and_updatedb
import requests
from truckms.service.model import create_session, VideoStatuses
import os
import GPUtil


def evaluate_workload():
    """
    Returns a number between 0 and 1 that signifies the workload on the current PC. 0 represents no workload,
    1 means no more work can be done efficiently.
    """
    deviceID = GPUtil.getFirstAvailable(order='first', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900,
                                        verbose=False)
    if len(deviceID) != 0:
        return True
    else:
        return False


def select_lru_worker():
    """
    Selects the least recently used worker from the known states and returns its IP and PORT
    """
    res1 = requests.get('http://localhost:5000/node_states').json()  # will get the data defined above
    res1 = [item for item in res1 if 'worker' in item['node_type'] or 'broker' in item['node_type']]
    if len(res1) == 0:
        print ("No worker or broker available")
        assert False
    res1 = sorted(res1, key=lambda x: x['workload'])
    return res1[0]['ip'], res1[0]['port']


def get_job_dispathcher(db_url, num_workers, max_operating_res, skip, analysis_func=None):
    """
    Creates a function that is able to dispatch work. Work can be done locally or remote.

    Args:
        db_url: url for database. used to store information about the received video files
        num_workers: how many concurrent jobs should be done locally before dispatching to a remote worker
        max_operating_res: operating resolution. bigger resolution will yield better detections
        skip: how many frames should be skipped when processing, recommended 0
        analysis_func: OPTIONAL.
    Return:
        function that can be called with a video_path
    """
    worker_pool = multiprocessing.Pool(num_workers)
    list_futures = []  # todo this list_futures should be removed. future responses should stay in database if are needed

    if analysis_func is None:
        analysis_func = partial(analyze_movie, max_operating_res=max_operating_res, skip=skip)

    def dispatch_work(video_path):
        if evaluate_workload():
            # do not remove this. this is useful. we don't want to upload in broker (waste time and storage when we want to process locally
            res = worker_pool.apply_async(func=analyze_and_updatedb, args=(db_url, video_path, analysis_func))
            list_futures.append(res)
        else:
            lru_ip, lru_port = select_lru_worker()
            session = create_session(db_url)
            VideoStatuses.add_video_status(session, file_path=video_path, results_path=None, remote_ip=lru_ip, remote_port=lru_port)
            data = {"max_operating_res": max_operating_res, "skip": skip}
            files = {os.path.basename(video_path): open(video_path, 'rb')}
            res = requests.post('http://{}:{}/upload_recordings'.format(lru_ip, lru_port), data=data, files=files)
            assert res.content == b"Files uploaded and started runniing the detector. Check later for the results"
            session.close()
            # do work remotely

    return dispatch_work, worker_pool, list_futures
