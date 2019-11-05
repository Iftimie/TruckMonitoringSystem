from truckms.service.service import create_microservice
import os.path as osp
import json
import pytest


@pytest.mark.skip(reason="takes too long on circleci")
def test_create_microservice(tmpdir):

    up_dir = osp.join(tmpdir.strpath, 'up_dir')
    app = create_microservice(up_dir, max_operating_res=320)
    assert osp.exists(up_dir)

    client = app.test_client()
    # aau-rainsnow dataset. found on kaggle
    data = {'cut.mkv': open(osp.join(osp.dirname(__file__), 'data', 'cut.mkv'), 'rb')}

    res = client.post("/upload_recordings", data=data)
    assert (res.status_code == 302) # 302 is found redirect
    # ret = json.loads(res.data)
    # assert "message" in ret
    # assert (ret["message"] == 'files uploaded successfully and started analyzing')

    assert osp.exists(osp.join(tmpdir.strpath, 'up_dir', 'cut.mkv'))
    app.worker_pool.close()
    app.worker_pool.join()
    assert osp.exists(osp.join(tmpdir.strpath, 'up_dir', 'cut.csv'))



