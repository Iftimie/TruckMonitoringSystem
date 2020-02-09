from truckms.service_v2.clientworker.p2p_clientworker import create_p2p_clientworker_app
from examples.newapi2.function import analyze_movie
import os
import shutil


if __name__ == "__main__":
    clientworker_app = create_p2p_clientworker_app("network_discovery_clientworker.txt", local_port=5002)
    path = '/home/achellaris/projects_data/TruckMonitoringSystem/service/clientworker.db'
    path = r'D:\tms_data\node_dirs\clientworker.db'

    if True:
        if os.path.exists(path):
            shutil.rmtree(path)

    clientworker_app.register_p2p_func(path, can_do_work_func=lambda: True)(analyze_movie)
    clientworker_app.run(host='0.0.0.0')
