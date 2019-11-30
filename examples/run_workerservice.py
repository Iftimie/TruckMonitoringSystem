def workermain():
    from truckms.service.worker.server import create_worker_service
    from truckms.service.bookkeeper import create_bookkeeper_p2pblueprint
    import os

    db_url = 'sqlite:///' + 'database.sqlite'
    up_dir = "/data1/workspaces/aiftimie/tms/worker_updir"
    remove_db = True
    if remove_db and os.path.exists(db_url.replace('sqlite:///', '')):
        os.remove(db_url.replace('sqlite:///', ''))

    port = 5000
    host = "0.0.0.0"

    app = create_worker_service(up_dir, db_url, 1)

    bookkeeper_bp = create_bookkeeper_p2pblueprint(local_port=port, app_roles=app.roles, discovery_ips_file="discovery_ips")
    app.register_blueprint(bookkeeper_bp)

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
    workermain()