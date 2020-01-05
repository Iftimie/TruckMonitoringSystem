from truckms.inference.neural import create_model, compute, pred_iter_to_pandas, pandas_to_pred_iter, plot_detections
from truckms.inference.analytics import filter_pred_detections
from truckms.inference.utils import framedatapoint_generator, framedatapoint_generator_by_frame_ids2
from truckms.inference.utils import create_avi
from itertools import tee
import os
import pandas as pd
import numpy as np
import os.path as osp
from truckms.api import PredictionDatapoint

def get_raw_df_from_movie(movie_path, model):
    g1 = framedatapoint_generator(movie_path, skip=0, max_frames=200)
    g2 = compute(g1, model, filter_classes=['train', 'truck', 'bus'])
    df = pred_iter_to_pandas(g2)
    return df


def get_tracked_df_from_df(df):
    g1 = filter_pred_detections(pandas_to_pred_iter(df))
    filtered_df = pred_iter_to_pandas(g1)
    return filtered_df


def get_common_frames(dataframes):
    kernel_size=10
    commond_ids = set()
    for df in dataframes:
        # TODO maybe apply a convolution to expand the frame ids
        selector = ~df["label"].isnull() | ~df.get('reason', pd.Series(index=df.index)).isnull()
        selector = (np.convolve(selector, [1] * kernel_size, 'same') > 0).astype(np.bool)
        commond_ids = commond_ids.union(set(df["img_id"][selector].tolist()))
    return sorted(commond_ids)


def normalize_dataframes(dataframes, common_frames):
    new_dataframes = []
    for df in dataframes:
        newdf = df.set_index("img_id").loc[common_frames].sort_index().reset_index()
        new_dataframes.append(newdf)
    return new_dataframes


def compare_multiple_dataframes(video_path, destination, *dataframes):
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


def compare_raw_vs_filtered(video_file):
    model = create_model()
    if not os.path.exists("df_raw.csv"):
        df_raw = get_raw_df_from_movie(video_file, model)
        df_raw.to_csv("df_raw.csv")
    else:
        df_raw = pd.read_csv("df_raw.csv")
    df_fil = get_tracked_df_from_df(df_raw)
    compare_multiple_dataframes(video_file, video_file.replace(':', '_').replace('\\', '_'), df_raw, df_fil)


def main():
    video_file = osp.join(osp.dirname(__file__),'..', '4K Traffic camera video - free download now!-MNn9qKG2UFI.webm')
    compare_raw_vs_filtered(video_file)

if __name__ == "__main__":
    main()