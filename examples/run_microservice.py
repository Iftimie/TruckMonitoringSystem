from truckms.service.service import create_microservice
from flaskwebgui import FlaskUI  # get the FlaskUI class
from truckms.service.worker.client import get_job_dispathcher


def flaskuimain():
    db_url = 'sqlite:///' + 'database.sqlite'
    work_func = get_job_dispathcher(db_url=db_url, num_workers=1, max_operating_res=320, skip=0)
    app = create_microservice(db_url, work_func)
    app = FlaskUI(app)
    app.run()


if __name__ == '__main__':
    flaskuimain()