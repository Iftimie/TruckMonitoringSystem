
def flaskuimain():
    from truckms.service.service import create_microservice
    from flaskwebgui import FlaskUI  # get the FlaskUI class
    from truckms.service.worker.user_client import get_job_dispathcher

    # to package app
    # https://www.reddit.com/r/Python/comments/bzql1t/create_htmlcssjavascript_gui_using_pythonflask/
    # https://github.com/ClimenteA/flaskwebgui

    db_url = 'sqlite:///' + 'database.sqlite'
    work_func, work_pool, list_futures = get_job_dispathcher(db_url=db_url, num_workers=1, max_operating_res=320, skip=0)
    app = create_microservice(db_url, work_func)
    app = FlaskUI(app)
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
