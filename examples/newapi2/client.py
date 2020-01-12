from truckms.service_v2.userclient.p2p_client import create_p2p_client_app
from examples.newapi2.function import analyze_movie, progress_hook
# from examples.evaluation.compare_motionmap_vs_everyframe import
from truckms.inference.analytics import filter_pred_detections
from truckms.inference.motion_map import movement_frames_indexes
import os

if __name__ == "__main__":
    # TODO save to database default arguments. currently this is not suported
    db_url = os.path.join(r"D:\tms_data\service", 'client.db')
    client_app = create_p2p_client_app("discovery_ips_client.txt")
    analyze_movie = client_app.register_p2p_func(db_url, "tms", "movie_statuses", can_do_locally_func=lambda :True)(analyze_movie)
    video_path = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'truckms', 'service', 'data', 'cut.mkv')
    video_handle = open(video_path, 'rb')
    res = analyze_movie(identifier="example1", video_handle=video_handle, progress_hook=progress_hook,
                  select_frame_inds_func=movement_frames_indexes,
                        filter_pred_detections_generator=filter_pred_detections)
    client_app.worker_pool.close()
    client_app.worker_pool.join()

    res = analyze_movie(identifier="example1", video_handle=video_handle, progress_hook=progress_hook,
                  select_frame_inds_func=movement_frames_indexes,
                        filter_pred_detections_generator=filter_pred_detections)
    print(res)