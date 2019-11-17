import threading
from threading import Thread
import time


def time_regular(list_funcs, time_interval):
    while True:
        time.sleep(time_interval)
        for f in list_funcs:
            f()


def flaskuimain():
    from truckms.service.service import create_guiservice, open_browser_func
    from flaskwebgui import FlaskUI  # get the FlaskUI class
    from truckms.service.worker.user_client import get_job_dispathcher
    from truckms.service.bookkeeper import create_bookkeeper_app
    from functools import partial

    # to package app
    # https://www.reddit.com/r/Python/comments/bzql1t/create_htmlcssjavascript_gui_using_pythonflask/
    # https://github.com/ClimenteA/flaskwebgui

    db_url = 'sqlite:///' + 'database.sqlite'

    port = 5000
    # host = "127.0.0.1"
    host = "0.0.0.0"
    time_interval = 10

    work_func, work_pool, list_futures = get_job_dispathcher(db_url=db_url, num_workers=1, max_operating_res=320, skip=0)
    app = create_guiservice(db_url, work_func)

    # TODO refactor app.roles, time_regular_funcs into a class

    app.roles = []
    app.time_regular_funcs = []

    bookkeeper_bp, bookkeeper_time_regular_func = create_bookkeeper_app(local_port=port, app_roles=app.roles, discovery_ips_file="discovery_ips")
    app.register_blueprint(bookkeeper_bp)
    app.time_regular_funcs.append(bookkeeper_time_regular_func)
    app.roles.append(bookkeeper_bp.role) # TODO maybe this one should not stay here

    thread1 = threading.Thread(target=time_regular, args=(app.time_regular_funcs, time_interval))
    thread1.start()

    app = FlaskUI(app, host=host, port=port)
    app.browser_thread = Thread(target=partial(open_browser_func, app, localhost='http://127.0.0.1:{}/'.format(port)))
    app.run()


if __name__ == '__main__':
    import logging.config
    import sys
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler('log.txt'),
                            logging.StreamHandler(sys.stdout)
                        ])
    flaskuimain()
