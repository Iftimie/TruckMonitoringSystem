from truckms.service.worker.broker import create_broker_microservice
import os.path as osp
import os


def test_upload_recordings(tmpdir):
    updir = osp.join(tmpdir.strpath, "updir")
    dburl = 'sqlite:///' + os.path.join(tmpdir.strpath, "video_statuses.db")
    os.mkdir(updir)
    app, worker_pool = create_broker_microservice(updir, dburl)
    client = app.test_client()
    with open(os.path.join(tmpdir.strpath, 'dummy.avi'), "w") as f: f.write("dummy_content")
    file_data = {'dummy.avi': open(os.path.join(tmpdir.strpath, 'dummy.avi'), 'rb')}
    json_data = {'max_operating_res': 320, 'skip': 0}
    file_data.update(json_data)
    client.post()

