def workerclientmain():
    from truckms.service.bookkeeper import create_bookkeeper_service
    from truckms.service.worker.worker_client import do_work
    from truckms.service.common import start_update_thread
    from functools import partial

    db_url = 'sqlite:///' + 'database.sqlite'
    up_dir = "/data1/workspaces/aiftimie/tms/worker_updir"

    port = 5000
    host = "0.0.0.0"
    time_interval = 10

    app = create_bookkeeper_service(local_port=port, discovery_ips_file="discovery_ips")
    app.time_regular_funcs.append(partial(do_work, up_dir, db_url, port))

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
    workerclientmain()