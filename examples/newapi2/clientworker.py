from truckms.service_v2.clientworker.p2p_clientworker import create_p2p_clientworker_app
from examples.newapi2.function import analyze_movie
import os
from truckms.service_v2.api import P2PFlaskApp


if __name__ == "__main__":
    clientworker_app = create_p2p_clientworker_app("discovery_ips_clientworker.txt",
                                                   P2PFlaskApp(__name__, local_port=5002))
    analyze_movie = clientworker_app.register_p2p_func(os.path.join(r"D:\tms_data\service", 'clientworker.db'),
                                                       "tms",
                                                       "movie_statuses")(analyze_movie)
    clientworker_app.register_time_regular_func(analyze_movie)
    clientworker_app.run(host='0.0.0.0')
