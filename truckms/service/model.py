from sqlalchemy import Column, Integer, String
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
Base = declarative_base()


class VideoStatuses(Base):
    """
    ORM for keeping account of video files submitted for processing
    """
    __tablename__ = 'video_statuses'
    id = Column(Integer, primary_key=True)
    file_path = Column(String, index=True)
    results_path = Column(String, nullable=True)

    @staticmethod
    def get_video_statuses(session):
        """
        Returns an iterable of items containing statuses of video files.
        Each item is an instance of VideoStatus and can access id, file_path and results_path
        """
        result = session.query(VideoStatuses).all()
        return result

    @staticmethod
    def add_video_status(session, **kwargs):
        """
        Args:
            session: sql session
            kwargs: keyword arguments used to assign values to the columns
        """
        session.add(VideoStatuses(**kwargs))
        session.commit()

    @staticmethod
    def update_results_path(session, file_path, new_results_path):
        """
        Updates the results_path column in VideoStatuses. This is when the detector has finished analyzing the video
        """
        query = session.query(VideoStatuses).filter_by(file_path=file_path, results_path=None).all()
        assert len(query) == 1
        item = query[0]
        item.results_path = new_results_path
        session.commit()


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
