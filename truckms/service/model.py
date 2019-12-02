from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import requests
import os
Base = declarative_base()


class HeartBeats(Base):
    """
    ORM for keeping account of heartbeats received by a broker
    """
    __tablename__ = 'heartbeats'
    id = Column(Integer, primary_key=True)
    time_of_heartbeat = Column(DateTime, default=datetime.now)

    @staticmethod
    def has_recent_heartbeat(session, minutes):
        """
        Returns true if there is a heartbeat less than x seconds ago
        """
        past = datetime.now() - timedelta(minutes=minutes)
        result = session.query(HeartBeats).all()
        for r in result:
            if r.time_of_heartbeat > past:
                return True
        return False

    @staticmethod
    def add_heartbeat(session, **kwargs):
        """
        Args:
            session: sql session
            kwargs: keyword arguments used to assign values to the columns
        """
        session.add(HeartBeats(**kwargs))
        session.commit()


class VideoStatuses(Base):
    """
    ORM for keeping account of video files submitted for processing

    file_path: local path to the video file
    results_path: local path to the results file
    remote_ip, remote_port: if the file cannot be processed locally it will be processed remotely
    """
    __tablename__ = 'video_statuses'
    id = Column(Integer, primary_key=True)
    file_path = Column(String, index=True)
    results_path = Column(String, nullable=True)
    remote_ip = Column(String, nullable=True)
    remote_port = Column(Integer, nullable=True)
    max_operating_res = Column(Integer, nullable=True)
    skip = Column(Integer, nullable=True)
    time_of_request = Column(DateTime, default=datetime.utcnow, nullable=True)
    progress = Column(Float, nullable=True, default=0.0)

    @staticmethod
    def get_video_statuses(session):
        """
        Returns an iterable of items containing statuses of video files.
        Each item is an instance of VideoStatus and can access id, file_path and results_path
        """

        result = session.query(VideoStatuses).all()
        return result

    # get video files that are remote and aren't finished in order to download the results
    @staticmethod
    def check_and_download(session):
        """
        This should be called after call to remove_dead_requests. Although this is not entirely necessary
        """
        query = session.query(VideoStatuses).filter(VideoStatuses.results_path == None,
                                                    VideoStatuses.remote_ip != None,
                                                    VideoStatuses.remote_port != None).all()
        for q in query:
            try:
                filename = os.path.basename(q.file_path)
                request_data = {"filename": filename}
                res = requests.get('http://{}:{}/download_results'.format(q.remote_ip, q.remote_port),
                                   data=request_data)
                if res.status_code == 200 and res.content != b'There is no file with this name: ' + bytes(filename,
                                                                                  encoding='utf8'):
                    filepath, _ = os.path.splitext(q.file_path)
                    q.results_path = filepath + ".csv"
                    with open(q.results_path, 'wb') as f:
                        f.write(res.content)
                    session.commit()
            except: # timeout error
                pass

    @staticmethod
    def remove_dead_requests(session):
        query = session.query(VideoStatuses).filter(VideoStatuses.results_path == None,
                                                    VideoStatuses.remote_ip != None,
                                                    VideoStatuses.remote_port != None).all()
        items_to_remove = []

        # TODO if server will stop or change IP while processing, then, if no results are returned X hours, then, a
        #  new request should be made
        for q in query:
            try:
                # TODO
                # before deleting, just check how old it its
                filename = os.path.basename(q.file_path)
                request_data = {"filename": filename}
                res = requests.get('http://{}:{}/download_results'.format(q.remote_ip, q.remote_port), data=request_data)
                if res.status_code == 404 and res.content == b'There is no file with this name: ' + bytes(filename,
                                                                                                          encoding='utf8'):
                    items_to_remove.append(q)
            except: # except timeout error
                items_to_remove.append(q)
        for item in items_to_remove:
            session.delete(item)
        session.commit()

    @staticmethod
    def add_video_status(session, **kwargs):
        """
        Args:
            session: sql session
            kwargs: keyword arguments used to assign values to the columns

        Return:
            return the newly added video status
        """
        vs = VideoStatuses(**kwargs)
        session.add(vs)
        session.commit()
        return vs


    @staticmethod
    def update_results_path(session, file_path, new_results_path):
        """
        Updates the results_path column in VideoStatuses. This is when the detector has finished analyzing the video
        """
        assert file_path is not None or new_results_path is not None
        if file_path is not None:
            query = session.query(VideoStatuses).filter_by(file_path=file_path, results_path=None).all()
        else:
            basename = os.path.splitext(os.path.basename(new_results_path))[0]
            query = session.query(VideoStatuses).filter(VideoStatuses.file_path.contains(basename)).all()
        assert len(query) == 1
        item = query[0]
        item.results_path = new_results_path
        session.commit()

    @staticmethod
    def find_results_path(session, file_path):
        """
        Finds the results path given a file_path
        """
        query = session.query(VideoStatuses).filter_by(file_path=file_path).all()
        #TODO in debug mode, because of multiple runs, more duplicates may appear
        assert len(query) >= 1
        return query[0]


def create_session(db_url):
    """
    Creates a session based on a db_url.

    Args:
        db_url: url to database. For example 'sqlite:///sales.db'
    """
    engine = create_engine(db_url, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    Base.metadata.create_all(engine)

    return session
