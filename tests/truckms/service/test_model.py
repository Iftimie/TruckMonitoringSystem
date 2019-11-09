from truckms.service.model import create_session
from truckms.service.model import VideoStatuses
import os

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
