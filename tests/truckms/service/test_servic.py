from truckms.service.service import create_microservice
from truckms.service.worker.client import get_job_dispathcher
import os.path as osp
from mock import Mock
from truckms.service import service
from truckms.service import worker
from truckms.service.model import create_session, VideoStatuses

# client.analyze_movie()
def test_new_microservice(tmpdir):
    service.gui_select_file = Mock(return_value="dummy_filename")
    worker.client.analyze_movie = Mock(return_value="dummy_results.csv")
    # client.analyze_movie =
    up_dir = osp.join(tmpdir.strpath, 'up_dir')
    db_url = 'sqlite:///' + osp.join(tmpdir.strpath, 'database.sqlite')
    work_func, worker_pool, list_futures = get_job_dispathcher(db_url=db_url, num_workers=1, max_operating_res=320, skip=0)
    app = create_microservice(up_dir, dispatch_work_func=work_func)
    client = app.test_client()
    res = client.get("/file_select")
    assert (res.status_code == 302)  # 302 is found redirect
    assert res.location == 'http://localhost/check_status'
    list_futures[0].get()
    worker_pool.close()
    worker_pool.join()
    session = create_session(db_url)
    query = VideoStatuses.get_video_statuses(session)
    assert len(query) == 1
    assert query[0].file_path == "dummy_filename"
    assert query[0].results_path == "dummy_results.csv"


