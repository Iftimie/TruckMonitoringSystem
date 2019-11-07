import os.path as osp
from truckms.service.service import create_microservice
from flaskwebgui import FlaskUI  # get the FlaskUI class


def main():
    up_dir = 'up_dir'
    app = create_microservice(up_dir, max_operating_res=800, skip=0)
    app.run(host="0.0.0.0", port=5000)


def flaskuimain():
    up_dir = 'up_dir'
    app = create_microservice(up_dir, max_operating_res=800, skip=0)
    app = FlaskUI(app)
    app.run()


if __name__ == '__main__':
    # main()
    flaskuimain()