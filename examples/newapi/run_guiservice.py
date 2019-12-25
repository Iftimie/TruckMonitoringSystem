def flaskuimain():
    from truckms.service_v2.userclient import create_guiservice
    from truckms.service.bookkeeper import create_bookkeeper_p2pblueprint
    import os

    db_url = 'sqlite:///' + r'D:\tms_data\guiservice\database.sqlite'
    remove_db = True
    if remove_db and os.path.exists(db_url.replace('sqlite:///', '')):
        os.remove(db_url.replace('sqlite:///', ''))

    port = 5000

    uiapp, app = create_guiservice(db_url, dispatch_work_func=lambda :None, port=port)

    bookkeeper_bp = create_bookkeeper_p2pblueprint(local_port=port, app_roles=app.roles, discovery_ips_file="discovery_ips")
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
