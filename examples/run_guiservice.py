from truckms.service.common import start_update_thread


def flaskuimain():
    from truckms.service.gui_interface import create_guiservice, open_browser_func
    from truckms.service.worker.user_client import get_job_dispathcher
    from truckms.service.bookkeeper import create_bookkeeper_app

    # to package app
    # https://www.reddit.com/r/Python/comments/bzql1t/create_htmlcssjavascript_gui_using_pythonflask/
    # https://github.com/ClimenteA/flaskwebgui

    db_url = 'sqlite:///' + 'database.sqlite'

    port = 5000
    time_interval = 10

    work_func, work_pool, list_futures = get_job_dispathcher(db_url=db_url, num_workers=1, max_operating_res=320, skip=0, local_port=port)
    uiapp, app = create_guiservice(db_url, work_func, port)

    # TODO refactor app.roles, time_regular_funcs into a class

    app.roles = []
    app.time_regular_funcs = []

    bookkeeper_bp, bookkeeper_time_regular_func = create_bookkeeper_app(local_port=port, app_roles=app.roles, discovery_ips_file="discovery_ips")
    app.register_blueprint(bookkeeper_bp)
    app.time_regular_funcs.append(bookkeeper_time_regular_func)
    # app.roles.append(bookkeeper_bp.role) # TODO maybe this one should not stay here

    start_update_thread(app.time_regular_funcs, time_interval)

    uiapp.run()


if __name__ == '__main__':
    import logging.config
    import sys
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler('log.txt'),
                            logging.StreamHandler(sys.stdout)
                        ])
    flaskuimain()
