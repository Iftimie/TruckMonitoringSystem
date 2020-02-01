def workerclientmain():
    from truckms.service.bookkeeper import create_bookkeeper_service
    from truckms.service.worker.worker_client import do_work
    from functools import partial
    import os

    stuff_dir = r"D:\tms_data\node_dirs\worker_client" # local
    stuff_dir = r"/data1/workspaces/aiftimie/tms/worker_updir" # local

    db_url = 'sqlite:///' + os.path.join(stuff_dir, 'database.sqlite')
    up_dir = r"D:\tms_data\node_dirs\worker_client\up_dir"  # local
    up_dir = r"/data1/workspaces/aiftimie/tms/worker_updir"  # server
    remove_db = True
    if remove_db and os.path.exists(db_url.replace('sqlite:///', '')):
        os.remove(db_url.replace('sqlite:///', ''))

    port = 5002
    host = "0.0.0.0"

    app = create_bookkeeper_service(local_port=port, discovery_ips_file="discovery_ips_client.txt")
    app.register_time_regular_func(partial(do_work, up_dir, db_url, port))

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
    workerclientmain()