import multiprocessing
from truckms.service.worker.server import analyze_movie
from truckms.service.model import create_session


def evaluate_workload():
    """
    Returns a number between 0 and 1 that signifies the workload on the current PC. 0 represents no workload,
    1 means no more work can be done efficiently.
    """
    return 0


def get_job_dispathcher(db_url, num_workers, max_operating_res, skip):
    """
    Creates a function that is able to dispatch work. Work can be done locally or remote.

    Args:
        db_url: url for database. used to store information about the received video files
        num_workers: how many concurrent jobs should be done locally before dispatching to a remote worker
        max_operating_res: operating resolution. bigger resolution will yield better detections
        skip: how many frames should be skipped when processing, recommended 0
    Return:
        function that can be called with a video_path
    """
    worker_pool = multiprocessing.Pool(num_workers)
    session = create_session(db_url)

    def dispatch_work(video_path):
        return
        if evaluate_workload() < 0.5:
            destination = analyze_movie(video_path, max_operating_res, skip)
        else:
            pass
            # do work remotely

    return dispatch_work