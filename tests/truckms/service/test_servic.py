from truckms.service.service import create_microservice
from truckms.service.worker.client import get_job_dispathcher
import os.path as osp
from mock import Mock
from truckms.service import service
from truckms.service.model import create_session, VideoStatuses
from truckms.service import worker

# in order for the thread to pickle the mock object I need to do this
# from pickle import dumps, loads
# loads(dumps(analyze_movie))
class PickableMock(Mock):
    def __init__(self):
        super(Mock, self).__init__()
        self.return_value = "dummy_results.csv"

    def __reduce__(self):
        return (PickableMock, ())

def test_new_microservice(tmpdir):
    service.gui_select_file = Mock(return_value="dummy_filename")
    worker.client.analyze_movie = PickableMock()

    db_url = 'sqlite:///' + osp.join(tmpdir.strpath, 'database.sqlite')
    work_func, worker_pool, list_futures = get_job_dispathcher(db_url=db_url, num_workers=1, max_operating_res=320, skip=0)
    app = create_microservice(db_url, dispatch_work_func=work_func)
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


def test_execution(tmpdir):
    input_file = osp.join(osp.dirname(__file__), 'data', 'cut.mkv')
    service.gui_select_file = Mock(return_value=input_file)

    db_url = 'sqlite:///' + osp.join(tmpdir.strpath, 'database.sqlite')
    work_func, worker_pool, list_futures = get_job_dispathcher(db_url=db_url, num_workers=1, max_operating_res=320, skip=0)
    app = create_microservice(db_url, dispatch_work_func=work_func)
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


