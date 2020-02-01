from truckms.service_v2.userclient.p2p_client import create_p2p_client_app
from examples.newapi2.function import analyze_movie, progress_hook
from truckms.inference.analytics import filter_pred_detections
from truckms.inference.motion_map import movement_frames_indexes
import os
from truckms.service_v2.api import P2PFlaskApp

import shutil



if __name__ == "__main__":
    client_app = create_p2p_client_app("discovery_ips_client.txt",
                                       P2PFlaskApp(__name__, local_port=5000))
    path = '/home/achellaris/projects_data/TruckMonitoringSystem/service/client.db'

    if True:
        if os.path.exists(path):
            shutil.rmtree(path)

    analyze_movie = client_app.register_p2p_func(path,
                                                 can_do_locally_func=lambda :False)(analyze_movie)

    res = analyze_movie(identifier="example1",
                        video_handle=open(os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'truckms', 'service', 'data', 'cut.mkv'), 'rb'),
                        progress_hook=progress_hook,
                        select_frame_inds_func=movement_frames_indexes,
                        filter_pred_detections_generator=filter_pred_detections)

    print(res.get())
