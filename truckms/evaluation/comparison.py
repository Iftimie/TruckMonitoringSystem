from itertools import tee
import numpy as np
import pandas as pd
from truckms.inference.neural import pandas_to_pred_iter, plot_detections
from truckms.inference.utils import framedatapoint_generator_by_frame_ids2, create_avi


def get_common_frames(dataframes):
    """
    Returns the sorted list of frame ids that is the union to all dataframes.
    A frame id id taken into consideration if the value for column "label" is not Null or "reason" is not Null.
    """
    kernel_size=10
    commond_ids = set()
    for df in dataframes:
        # df.get('reason', pd.Series(index=df.index)) will create the column if it does not exist. TODO remove in future
        selector = ~df["label"].isnull() | ~df.get('reason', pd.Series(index=df.index)).isnull()
        selector = (np.convolve(selector, [1] * kernel_size, 'same') > 0).astype(np.bool)
        commond_ids = commond_ids.union(set(df["img_id"][selector].tolist()))
    return sorted(commond_ids)


def normalize_dataframes(dataframes, common_frames):
    """
    Some dataframes might not contain all frame ids that are present in common_frames. No problem. Just insert the
        required rows
    """
    new_dataframes = []
    for df in dataframes:
        newdf = df.set_index("img_id").loc[common_frames].sort_index().reset_index()
        new_dataframes.append(newdf)
    return new_dataframes


def compare_multiple_dataframes(video_path, destination, *dataframes):
    """
    Given a set of dataframes this function will look at the union of the dataframes. The resulted frames will be the
    union of the dataframes where there is a prediction or there is a reason (motionmap) for the presence of a frame.
    In other words, rows that have at least one column not Null.

    In this way you can see frames where one detector managed to detect something when the other didn't.

    The dataframes do not need to have the same number of rows in order to be compared. The presence of the 'img_id'
    column is sufficient.

    Args:
        video_path: video file from which the dataframes resulted
        destination: path to a destination .mp4 or .avi file that will contain the plotting of the dataframes
        dataframes: positional arguments with dataframes

    Result:
        None
    """
    common_frames = get_common_frames(dataframes)
    dataframes = normalize_dataframes(dataframes, common_frames)
    g = framedatapoint_generator_by_frame_ids2(video_path, common_frames)
    fdpgs = tee(g, len(dataframes))

    gdfs = [pandas_to_pred_iter(df) for df in dataframes]
    plots = [plot_detections(fdpg, gdf) for fdpg, gdf in zip(fdpgs, gdfs)]
    pairsg = zip(*plots)

    fdp_list = next(pairsg)
    assert all(fdp.frame_id == fdp_list[0].frame_id for fdp in fdp_list)

    images = [fdp.image for fdp in fdp_list]
    frame = np.concatenate((*images,), axis=1)
    with create_avi(destination, frame) as append_fn:
        for fdp_list in pairsg:
            print(fdp_list)
            assert all(fdp.frame_id == fdp_list[0].frame_id for fdp in fdp_list)
            images = [fdp.image for fdp in fdp_list]
            frame = np.concatenate((*images,), axis=1)
            append_fn(frame)
