from truckms.service_v2.brokerworker.p2p_brokerworker import create_p2p_brokerworker_app
from examples.newapi2.function import analyze_movie, progress_hook
import os
from truckms.service_v2.api import P2PFlaskApp


if __name__ == "__main__":
    broker_worker_app = create_p2p_brokerworker_app("discovery_ips_brokerworker.txt",
                                                    P2PFlaskApp(__name__, local_port=5001))
    analyze_movie = broker_worker_app.register_p2p_func(os.path.join(r"D:\tms_data\service", 'brokerworker.db'),
                                                        "tms",
                                                        "movie_statuses",
                                                        can_do_locally_func=lambda :True,
                                                        time_limit=12)(analyze_movie)
    broker_worker_app.run(host='0.0.0.0')
