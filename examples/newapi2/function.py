from truckms.inference.neural import create_model, pred_iter_to_pandas, compute
from truckms.inference.neural import create_model_efficient
from truckms.inference.utils import framedatapoint_generator, get_video_file_size, framedatapoint_generator_by_frame_ids2
from collections import Callable
from typing import Iterable
from functools import partial
import os
from truckms.evaluation.comparison import compare_multiple_dataframes
from truckms.api import PredictionDatapoint
import io
from truckms.inference.analytics import filter_pred_detections
from truckms.inference.motion_map import movement_frames_indexes
from truckms.service_v2.userclient.p2p_client import p2p_progress_hook


def generator_hook(video_path, pdp_iter: Iterable[PredictionDatapoint], progress_hook: Callable) -> Iterable[
    PredictionDatapoint]:
    """
    Used in analyze_movie
    """
    size = get_video_file_size(video_path) - 1
    # size - 1 because we are working with indexes. if frame_id=size-1 then the percentage done is 100
    checkpoint = int((size - 1) * 5 / 100)
    for pdp in pdp_iter:
        if (pdp.frame_id + 1) % checkpoint == 0:
            progress_hook(pdp.frame_id, size)
        yield pdp

"""
For implementing callables in a safe way:
https://docs.python.org/2/library/pdb.html
https://blog.magrathealabs.com/filesystem-events-monitoring-with-python-9f5329b651c3
"""

def analyze_movie(identifier: str, video_handle: io.IOBase) -> {"results": io.IOBase, "video_results": io.IOBase}:
    """
    Args:
        identifier: identifier parameter to be compliant with the p2p framework
        video_handle: file object for the movie to be analyzed.
        progress_hook: function that is called with two integer arguments. the first one represents the current frame index
            the second represents the final index.
    Return:
          dictionary containing path to the .csv file and to .mp4 file
    """
    model = create_model_efficient(model_creation_func=partial(create_model, max_operating_res=320))

    video_handle.close()
    video_file = video_handle.name
    frame_ids = movement_frames_indexes(video_file)
    if frame_ids is None:
        image_gen = framedatapoint_generator(video_file, skip=0, reason=None)
    else:
        image_gen = framedatapoint_generator_by_frame_ids2(video_file, frame_ids, reason="motionmap")

    # TODO set batchsize by the available VRAM
    pred_gen = compute(image_gen, model=model, batch_size=5)
    filtered_pred = filter_pred_detections(pred_gen)
    if p2p_progress_hook is not None:
        filtered_pred = generator_hook(video_file, filtered_pred, p2p_progress_hook)
    df = pred_iter_to_pandas(filtered_pred)
    destination = os.path.splitext(video_file)[0]+'.csv'
    df.to_csv(destination)
    if p2p_progress_hook is not None:
        # call one more time the hook. this is just for clean ending of the processing. it may happen in case where the
        # skip is 5 that the final index is not reached, and in percentage it will look like 99.9% finished
        size = get_video_file_size(video_file) - 1
        p2p_progress_hook(size, size)

    visual_destination = os.path.splitext(video_file)[0]+'_results.avi'
    compare_multiple_dataframes(video_file,
                                visual_destination,
                                df)
    return {"results": open(destination, 'rb'),
            "video_results": open(visual_destination, 'rb')}
