from truckms.inference.neural import create_model, pred_iter_to_pandas, compute
from truckms.inference.neural import create_model_efficient
from truckms.inference.utils import framedatapoint_generator, get_video_file_size, framedatapoint_generator_by_frame_ids2
from collections import Callable
from typing import Iterable
from functools import partial
import os
from truckms.api import PredictionDatapoint
import io


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


def progress_hook(curidx, endidx) -> dict:
    return {"progress": curidx/endidx * 100}


def analyze_movie(identifier: str, video_handle: io.IOBase,
                  select_frame_inds_func: Callable,
                  progress_hook: Callable,
                  filter_pred_detections_generator: Callable) -> {"results": io.IOBase}:
    """
    Attention!!! if the movie is short or too fast and skip  is too big, then it may result with no detections
    #TODO think about this

    Args:
        video_path: path to a file on the local disk
        max_operating_res: the resolution used for analyzing the video file. The lower the resolution, the earlier it
            will finish but with less accuracy
        skip: number of frames to skip. the higher the number, the earlier the function will finish
        progress_hook: function that is called with two integer arguments. the first one represents the current frame index
            the second represents the final index. the function should not return anything. The hook will actually get
            called once every 5% is done of the total work
    Return:
          path to results file
    """
    model = create_model_efficient(model_creation_func=partial(create_model, max_operating_res=320))

    video_handle.close()
    video_file = video_handle.name
    frame_ids = select_frame_inds_func(video_file)
    if frame_ids is None:
        image_gen = framedatapoint_generator(video_file, skip=0, reason=None)
    else:
        image_gen = framedatapoint_generator_by_frame_ids2(video_file, frame_ids, reason="motionmap")

    # TODO set batchsize by the available VRAM
    pred_gen = compute(image_gen, model=model, batch_size=5)
    filtered_pred = filter_pred_detections_generator(pred_gen)
    if progress_hook is not None:
        filtered_pred = generator_hook(video_file, filtered_pred, progress_hook)
    df = pred_iter_to_pandas(filtered_pred)
    destination = os.path.splitext(video_file)[0]+'.csv'
    df.to_csv(destination)
    if progress_hook is not None:
        # call one more time the hook. this is just for clean ending of the processing. it may happen in case where the
        # skip is 5 that the final index is not reached, and in percentage it will look like 99.9% finished
        size = get_video_file_size(video_file) - 1
        progress_hook(size, size)

    return {"results": open(destination, 'rb')}