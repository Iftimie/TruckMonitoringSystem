from examples.newapi2.logging_config import configure_logger
from truckms.service_v2.api import P2PFlaskApp
from truckms.service_v2.userclient import p2p_client
from examples.newapi2.function import analyze_movie
import os
import shutil


if __name__ == "__main__":
    configure_logger("client", module_level_list=[(p2p_client, 'INFO')])
    # TODO this configuration could be moved into create_p2p_client_app

    client_app = p2p_client.create_p2p_client_app("network_discovery_client.txt", local_port=5000)
    path = '/home/achellaris/projects_data/TruckMonitoringSystem/service/client.db'
    path = r'D:\tms_data\node_dirs\client.db'

    if True:
        if os.path.exists(path):
            shutil.rmtree(path)

    analyze_movie = client_app.register_p2p_func(path,
                                                 can_do_locally_func=lambda: True)(analyze_movie)

    res = analyze_movie(video_handle=open(os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'truckms', 'service', 'data', 'cut.mkv'), 'rb'))

    print(res.get())

    client_app.background_thread.shutdown()
