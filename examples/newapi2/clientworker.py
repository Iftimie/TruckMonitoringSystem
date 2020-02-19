from truckms.service_v2.clientworker.p2p_clientworker import P2PClientworkerApp
from examples.newapi2.function import analyze_movie
import os
import shutil


if __name__ == "__main__":
    password = "super secret password"
    path = '/home/achellaris/projects_data/TruckMonitoringSystem/service/clientworker.db'

    if True:
        if os.path.exists(path):
            shutil.rmtree(path)
            while os.path.exists(path): pass

    clientworker_app = P2PClientworkerApp("network_discovery_clientworker.txt", local_port=5002, mongod_port=5102,
                                                   password=password, cache_path=path)
    # path = r'D:\tms_data\node_dirs\clientworker.db'

    clientworker_app.register_p2p_func(can_do_work_func=lambda: True)(analyze_movie)
    clientworker_app.run(host='0.0.0.0')
