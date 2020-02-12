from truckms.service_v2.brokerworker.p2p_brokerworker import create_p2p_brokerworker_app
from examples.newapi2.function import analyze_movie
import os
import shutil


if __name__ == "__main__":
    password = "super secret password"
    path = '/home/achellaris/projects_data/TruckMonitoringSystem/service/brokerworker.db'

    if True:
        if os.path.exists(path):
            shutil.rmtree(path)

    broker_worker_app = create_p2p_brokerworker_app("network_discovery_brokerworker.txt", local_port=5001,
                                                    password=password, cache_path=path)
    # path = r'D:\tms_data\node_dirs\brokerworker.db'


    analyze_movie = broker_worker_app.register_p2p_func(can_do_locally_func=lambda: True,
                                                        time_limit=12)(analyze_movie)
    broker_worker_app.run(host='0.0.0.0')
