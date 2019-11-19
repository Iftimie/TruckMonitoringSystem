def flaskuimain():
    from truckms.service.worker.server import create_worker_microservice
    from truckms.service.bookkeeper import create_bookkeeper_app
    from truckms.service.common import start_update_thread
    from functools import partial

    # to package app
    # https://www.reddit.com/r/Python/comments/bzql1t/create_htmlcssjavascript_gui_using_pythonflask/
    # https://github.com/ClimenteA/flaskwebgui

    db_url = 'sqlite:///' + 'database.sqlite'
    up_dir = "/data1/workspaces/aiftimie/tms/worker_updir"

    port = 5000
    host = "0.0.0.0"
    time_interval = 10

    app, worker_pool = create_worker_microservice(up_dir, db_url, 1)
    # app allready has roles after calling create_worker_microservice
    # also time_regular_funcs

    bookkeeper_bp, bookkeeper_time_regular_func = create_bookkeeper_app(local_port=port, app_roles=app.roles, discovery_ips_file="discovery_ips")
    app.register_blueprint(bookkeeper_bp)
    app.time_regular_funcs.append(bookkeeper_time_regular_func)
    # app.roles.append(bookkeeper_bp.role)  # TODO maybe this one should not stay here

    start_update_thread(app.time_regular_funcs, time_interval)

    app.run(host=host, port=port)


if __name__ == '__main__':
    import logging.config
    import sys
    import torch

    torch.multiprocessing.set_start_method('spawn')
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler('log.txt'),
                            logging.StreamHandler(sys.stdout)
                        ])
    flaskuimain()