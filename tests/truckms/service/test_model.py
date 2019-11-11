from truckms.service.model import create_session
from truckms.service.model import VideoStatuses
import os
import requests


def test_session(tmpdir):
    print (tmpdir)
    url = 'sqlite:///' + os.path.join(tmpdir.strpath, "video_statuses.db")
    session = create_session(url)
    c1 = VideoStatuses(file_path='blabla.avi', results_path='blabla.csv')
    session.add(c1)
    session.commit()
    results = VideoStatuses.get_video_statuses(session)
    assert len(results) == 1
    assert results[0].file_path == 'blabla.avi'
    assert results[0].results_path == 'blabla.csv'

    c2 = VideoStatuses(file_path='blabla.avi', results_path=None)
    session.add(c2)
    session.commit()
    results = VideoStatuses.get_video_statuses(session)
    assert len(results) == 2
    assert results[1].file_path == 'blabla.avi'
    assert results[1].results_path is None

    query = session.query(VideoStatuses).filter_by(file_path='blabla.avi', results_path=None)
    item = query[0]
    item.results_path = "new_blabla.csv"
    session.commit()

    results = VideoStatuses.get_video_statuses(session)
    assert len(results) == 2
    assert results[1].file_path == 'blabla.avi'
    assert results[1].results_path == 'new_blabla.csv'


def test_high_level_api(tmpdir):
    url = 'sqlite:///' + os.path.join(tmpdir.strpath, "video_statuses.db")
    session = create_session(url)
    VideoStatuses.add_video_status(session, file_path="blabla.avi", results_path=None)
    results = VideoStatuses.get_video_statuses(session)
    assert len(results) == 1
    VideoStatuses.update_results_path(session, file_path="blabla.avi", new_results_path="new_blabla.csv")
    results = VideoStatuses.get_video_statuses(session)
    assert len(results) == 1
    assert results[0].file_path == "blabla.avi"
    assert results[0].results_path == "new_blabla.csv"


def test_remote_columns(tmpdir):
    url = 'sqlite:///' + os.path.join(tmpdir.strpath, "video_statuses.db")
    session = create_session(url)
    VideoStatuses.add_video_status(session, file_path="blabla.avi", results_path=None, remote_ip="127.0.0.1", remote_port=5000)
    results = VideoStatuses.get_video_statuses(session)
    assert len(results) == 1


from truckms.service.worker.server import create_worker_microservice
from truckms.service.bookkeeper import ServerThread
def test_query_remote(tmpdir):
    url = 'sqlite:///' + os.path.join(tmpdir.strpath, "video_statuses.db")
    session = create_session(url)
    VideoStatuses.add_video_status(session, file_path="blabla.avi", results_path=None, remote_ip="127.0.0.1",
                                   remote_port=5000)
    results = VideoStatuses.get_video_statuses(session)
    assert len(results) == 1
    query = session.query(VideoStatuses).filter(VideoStatuses.results_path == None,
                                                VideoStatuses.remote_ip != None,
                                                VideoStatuses.remote_port != None).all()
    for q in query:
        try:
            res = requests.get('http://{}:{}/download_results'.format(q.remote_ip, q.remote_port), data=q.file_path)
            assert False # no server should be opened here
        except:
            assert True

    up_dir = os.path.join(tmpdir.strpath, "updir")
    os.mkdir(up_dir)
    worker_db_url = 'sqlite:///' + os.path.join(tmpdir.strpath, "workerdb.db")
    app, _ = create_worker_microservice(up_dir, db_url=worker_db_url,num_workers=1)
    server1 = ServerThread(app, port=5000)
    server1.start()
    for q in query:
        request_data = {"filename":q.file_path}
        res = requests.get('http://{}:{}/download_results'.format(q.remote_ip, q.remote_port), data=request_data)
        assert res.content == b'There is no file with this name: ' + bytes(q.file_path, encoding='utf8')
        assert res.status_code == 404

    server1.shutdown()
