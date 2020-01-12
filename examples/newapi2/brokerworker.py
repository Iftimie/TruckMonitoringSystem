from truckms.service_v2.brokerworker.p2p_brokerworker import create_p2p_brokerworker_app
from examples.newapi2.function import analyze_movie, progress_hook
# from examples.evaluation.compare_motionmap_vs_everyframe import
from truckms.inference.analytics import filter_pred_detections
from truckms.inference.motion_map import movement_frames_indexes
import os
from truckms.service_v2.api import P2PFlaskApp


if __name__ == "__main__":
    # TODO save to database default arguments. currently this is not suported
    db_url = os.path.join(r"D:\tms_data\service", 'brokerworker.db')
    broker_worker_app = P2PFlaskApp(__name__, local_port=5001)
    broker_worker_app = create_p2p_brokerworker_app("discovery_ips_brokerworker.txt", broker_worker_app)
    analyze_movie = broker_worker_app.register_p2p_func(db_url, "tms", "movie_statuses", can_do_locally_func=lambda :True)(analyze_movie)

    broker_worker_app.run()