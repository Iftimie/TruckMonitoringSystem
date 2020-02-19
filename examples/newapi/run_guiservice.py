def flaskuimain():
    from truckms.service_v2.userclient.userclient import create_guiservice, get_job_dispathcher
    from truckms.service_v2.api import create_bookkeeper_p2pblueprint
    import os

    db_url = r'D:\tms_data\guiservice\tinymongo.db'
    remove_db = True
    if remove_db and os.path.exists(os.path.join(db_url, 'tms.json')):
        os.remove(os.path.join(db_url, 'tms.json'))
        os.rmdir(db_url)

    port = 5000
    dispatch_work, _, _ = get_job_dispathcher(db_url, 1, port)
    uiapp, app = create_guiservice(db_url, dispatch_work_func=dispatch_work, port=port)

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
