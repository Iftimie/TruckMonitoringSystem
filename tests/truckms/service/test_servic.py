from truckms.service.service import create_microservice
from truckms.service.worker.client import get_job_dispathcher
import os.path as osp
from mock import Mock
from truckms.service import service


def test_new_microservice(tmpdir):
    service.gui_select_file = Mock(return_value="dummy_filename")
    up_dir = osp.join(tmpdir.strpath, 'up_dir')
    db_url = 'sqlite:///' + osp.join(tmpdir.strpath, 'database.sqlite')
    work_func = get_job_dispathcher(db_url=db_url, num_workers=1, max_operating_res=320, skip=0)
    app = create_microservice(up_dir, dispatch_work_func=work_func)
    client = app.test_client()
    res = client.get("/file_select")
    assert (res.status_code == 302)  # 302 is found redirect
    assert res.location == 'http://localhost/check_status'


