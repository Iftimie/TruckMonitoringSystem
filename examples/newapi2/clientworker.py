from truckms.service_v2.clientworker.p2p_clientworker import create_p2p_clientworker_app
from examples.newapi2.function import analyze_movie, progress_hook
# from examples.evaluation.compare_motionmap_vs_everyframe import
from truckms.inference.analytics import filter_pred_detections
from truckms.inference.motion_map import movement_frames_indexes
import os
from truckms.service_v2.api import P2PFlaskApp
import time

import threading
def fun1(app):
    app.run(host='0.0.0.0')

if __name__ == "__main__":
    # TODO save to database default arguments. currently this is not suported
    db_url = os.path.join(r"D:\tms_data\service", 'clientworker.db')
    clientworker_app = P2PFlaskApp(__name__, local_port=5002)
    clientworker_app = create_p2p_clientworker_app("discovery_ips_clientworker.txt", clientworker_app)
    analyze_movie = clientworker_app.register_p2p_func(db_url, "tms", "movie_statuses")(analyze_movie)

    thread1 = threading.Thread(target=fun1, args=(clientworker_app,))
    thread1.start()
    time.sleep(10)  # add event here instead of sleep

    analyze_movie()
    pass