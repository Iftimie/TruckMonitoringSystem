from truckms.service.worker.server import create_worker_p2pblueprint, create_worker_service, analyze_and_updatedb, \
    analyze_movie
import os
from mock import Mock
import os.path as osp
from truckms.service import worker
from truckms.service.model import create_session
from truckms.service.model import VideoStatuses
from functools import partial


class PickableMock(Mock):

    return_value = "dummy.csv"
    def __init__(self):
        super(Mock, self).__init__()
        self.return_value = PickableMock.return_value

    def __reduce__(self):
        return (PickableMock, ())

def test_create_worker_blueprint(tmpdir):
    worker.server.analyze_movie = PickableMock()

    up_dir = os.path.join(tmpdir.strpath, "updir")
    os.mkdir(up_dir)
    db_url = 'sqlite:///' + os.path.join(tmpdir.strpath, "database.sqlite")
    worker_app = create_worker_service(up_dir, db_url, 1)
    client = worker_app.test_client()

    with open(os.path.join(tmpdir.strpath, 'dummy.avi'), 'wb') as f: pass
    file_data = {'dummy.avi': open(os.path.join(tmpdir.strpath, 'dummy.avi'), 'rb')}
    json_data = {'max_operating_res': 320, 'skip': 0}
    file_data.update(json_data)
    res = client.post("/upload_recordings", data=file_data)
    assert (res.status_code == 200)
    assert (res.data == b"Files uploaded and started runniing the detector. Check later for the results")
    worker_app._blueprints['worker_bp'].worker_pool.close()
    worker_app._blueprints['worker_bp'].worker_pool.join()
    session = create_session(db_url)
    results = VideoStatuses.get_video_statuses(session)
    assert len(results) == 1
    assert len(results) == 1
    assert results[0].file_path == os.path.join(up_dir, 'dummy.avi')
    assert results[0].results_path == "dummy.csv"


def test_analyze_and_updatedb(tmpdir):
    db_url = 'sqlite:///' + os.path.join(tmpdir.strpath, "database.sqlite")
    test_video_path = osp.join(osp.dirname(__file__), '..', '..', 'service', 'data', 'cut.mkv')
    analyis_func = partial(analyze_movie, max_operating_res=320, skip=0)
    analyze_and_updatedb(db_url, test_video_path, analysis_func=analyis_func)
    session = create_session(db_url)
    statuses = VideoStatuses.get_video_statuses(session)
    assert statuses[0].progress == 100.0
