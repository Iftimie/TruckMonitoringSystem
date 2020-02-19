from truckms.service_v2.userclient.p2p_client import create_p2p_client_app
from examples.newapi2.function import analyze_movie
import os
import shutil


if __name__ == "__main__":

    password = "super secret password"
    path = '/home/achellaris/projects_data/TruckMonitoringSystem/service/client.db'

    if True:
        if os.path.exists(path):
            shutil.rmtree(path)
            while os.path.exists(path): pass

    client_app = create_p2p_client_app("network_discovery_client.txt", local_port=5000, mongod_port=5100, password=password, cache_path=path)
    # path = r'D:\tms_data\node_dirs\client.db'

    analyze_movie = client_app.register_p2p_func(can_do_locally_func=lambda: False)(analyze_movie)

    res = analyze_movie(video_handle=open(os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'truckms', 'service', 'data', 'cut.mkv'), 'rb'),
                        arg2=100)
    print(res.get())
    res = analyze_movie(video_handle=open(
        os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'truckms', 'service', 'data', 'cut.mkv'), 'rb'),
                        arg2=200)
    print(res.get())

    client_app.background_server.shutdown()
