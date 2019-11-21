from truckms.service.gui_interface import create_guiservice
from truckms.service.worker.user_client import get_job_dispathcher
from truckms.service.worker import user_client
import os.path as osp
from mock import Mock
from truckms.service import gui_interface
from truckms.service.model import create_session, VideoStatuses
from truckms.service import worker
import os
import pytest

# in order for the thread to pickle the mock object I need to do this
# from pickle import dumps, loads
# loads(dumps(analyze_movie))
class PickableMock(Mock):
    def __init__(self):
        super(Mock, self).__init__()
        self.return_value = "dummy_results.csv"

    def __reduce__(self):
        return (PickableMock, ())

# in succesive tests, the pickable mock persists, thus in test_execution, the new original function will be reasigned
original_function = worker.user_client.analyze_movie

def test_new_microservice(tmpdir):
    gui_interface.gui_select_file = Mock(return_value="dummy_filename")
    user_client.select_lru_worker = Mock(return_value=(None, None))

    worker.user_client.analyze_movie = PickableMock()

    db_url = 'sqlite:///' + osp.join(tmpdir.strpath, 'database.sqlite')
    work_func, worker_pool, list_futures = get_job_dispathcher(db_url=db_url, num_workers=1, max_operating_res=320, skip=0,
                                                               local_port=5000)
    uiapp, app = create_guiservice(db_url, dispatch_work_func=work_func, port=5000)
    client = app.test_client()
    res = client.get("/file_select")
    assert (res.status_code == 302)  # 302 is found redirect
    assert res.location == 'http://localhost/check_status'
    worker_pool.close()
    worker_pool.join()
    session = create_session(db_url)
    query = VideoStatuses.get_video_statuses(session)
    assert len(query) == 1
    assert query[0].file_path == "dummy_filename"
    assert query[0].results_path == "dummy_results.csv"


@pytest.mark.skip(reason="takes too long and in CircleCI it will timeout and fail the test")
def test_execution(tmpdir):
    input_file = osp.join(osp.dirname(__file__), 'data', 'cut.mkv')
    worker.user_client.analyze_movie = original_function

    gui_interface.gui_select_file = Mock(return_value=input_file)
    user_client.select_lru_worker = Mock(return_value=(None, None))

    db_url = 'sqlite:///' + osp.join(tmpdir.strpath, 'database.sqlite')
    if osp.exists(db_url.replace('sqlite:///', '')):
        os.remove(db_url.replace('sqlite:///', ''))
    work_func, worker_pool, list_futures = get_job_dispathcher(db_url=db_url, num_workers=1, max_operating_res=320, skip=0,
                                                               local_port=5000)
    uiapp, app = create_guiservice(db_url, dispatch_work_func=work_func, port=5000)
    client = app.test_client()
    res = client.get("/file_select")
    assert (res.status_code == 302)  # 302 is found redirect
    assert res.location == 'http://localhost/check_status'
    worker_pool.close()
    worker_pool.join()
    session = create_session(db_url)
    query = VideoStatuses.get_video_statuses(session)
    assert len(query) == 1
    assert query[0].file_path == input_file
    assert query[0].results_path == input_file.replace('mkv', 'csv')


