from truckms.service.worker.server import create_worker_blueprint, create_worker_microservice
import os


def test_create_worker_blueprint(tmpdir):
    up_dir = os.mkdir(os.path.join(tmpdir.strpath, "updir"))
    db_url = 'sqlite:///' + os.path.join(tmpdir.strpath, "database.sqlite")
    worker_app = create_worker_microservice(up_dir, db_url, 1)
