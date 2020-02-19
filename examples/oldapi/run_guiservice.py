def flaskuimain():
    from truckms.service.gui_interface import create_guiservice
    from truckms.service.worker.user_client import get_job_dispathcher
    from truckms.service_v2.api import create_bookkeeper_p2pblueprint
    import os
    # to package app
    # https://www.reddit.com/r/Python/comments/bzql1t/create_htmlcssjavascript_gui_using_pythonflask/
    # https://github.com/ClimenteA/flaskwebgui

    db_url = 'sqlite:///' + r'D:\tms_data\guiservice\database.sqlite'
    remove_db = True
    if remove_db and os.path.exists(db_url.replace('sqlite:///', '')):
        os.remove(db_url.replace('sqlite:///', ''))

    port = 5000

    work_func, work_pool, list_futures = get_job_dispathcher(db_url=db_url, num_workers=1, max_operating_res=320, skip=0, local_port=port)
    uiapp, app = create_guiservice(db_url, work_func, port)

    bookkeeper_bp = create_bookkeeper_p2pblueprint(local_port=port, app_roles=app.roles, discovery_ips_file="discovery_ips_client.txt")
    app.register_blueprint(bookkeeper_bp)

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
