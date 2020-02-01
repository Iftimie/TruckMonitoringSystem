def brokerworkermain():
    from truckms.service_v2.brokerworker.brokerworker import create_brokerworker_microservice
    from truckms.service.bookkeeper import create_bookkeeper_p2pblueprint
    import os

    up_dir = "/data1/workspaces/aiftimie/tms/worker_updir"
    up_dir = r"D:\tms_data\node_dirs\broker\up_dir"  # work
    db_url = os.path.join(up_dir, 'tinymongo.db')
    remove_db = True
    if remove_db and os.path.exists(os.path.join(db_url, 'tms.json')):
        os.remove(os.path.join(db_url, 'tms.json'))
        os.rmdir(db_url)

    port = 5001
    host = "0.0.0.0"

    app = create_brokerworker_microservice(up_dir, db_url, num_workers=0)

    bookkeeper_bp = create_bookkeeper_p2pblueprint(local_port=port, app_roles=app.roles, discovery_ips_file= "discovery_ips_client.txt")
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
    brokerworkermain()