from truckms.service_v2.userclient.p2p_client import create_p2p_client_app
from examples.newapi2.function import analyze_movie, progress_hook
from truckms.inference.analytics import filter_pred_detections
from truckms.inference.motion_map import movement_frames_indexes
import os
from truckms.service_v2.api import P2PFlaskApp
import time
import threading


def fun1(app):
    app.run(host='0.0.0.0')


if __name__ == "__main__":
    client_app = create_p2p_client_app("discovery_ips_client.txt",
                                       P2PFlaskApp(__name__, local_port=5000))
    analyze_movie = client_app.register_p2p_func(os.path.join(r"D:\tms_data\service", 'client.db'),
                                                 "tms",
                                                 "movie_statuses",
                                                 can_do_locally_func=lambda :False)(analyze_movie)

    threading.Thread(target=fun1, args=(client_app,)).start()
    time.sleep(5)  # add event here instead of sleep

    res = analyze_movie(identifier="example1",
                        video_handle=open(os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'truckms', 'service', 'data', 'cut.mkv'), 'rb'),
                        progress_hook=progress_hook,
                        select_frame_inds_func=movement_frames_indexes,
                        filter_pred_detections_generator=filter_pred_detections)

    print(res.get())
