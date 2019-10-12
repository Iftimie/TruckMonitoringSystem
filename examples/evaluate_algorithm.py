from truckms.inference.neural import create_model, compute, pred_iter_to_pandas, pandas_to_pred_iter, plot_detections
from truckms.inference.analytics import filter_pred_detections
from truckms.inference.utils import framedatapoint_generator
from truckms.inference.utils import create_avi
from itertools import tee
import os
import pandas as pd
import numpy as np

def get_raw_df_from_movie(movie_path, model):
    g1 = framedatapoint_generator(movie_path, skip=0, max_frames=200)
    g2 = compute(g1, model, filter_classes=['train', 'truck', 'bus'])
    df = pred_iter_to_pandas(g2)
    return df


def get_tracked_df_from_df(df):
    g1 = filter_pred_detections(pandas_to_pred_iter(df))
    filtered_df = pred_iter_to_pandas(g1)
    return filtered_df


def compare_two_dataframes(video_path, df1, df2, destination):
    g = framedatapoint_generator(video_path=video_path, skip=0)
    fdpg1, fdpg2 = tee(g)
    gdf1 = pandas_to_pred_iter(df1)
    gdf2 = pandas_to_pred_iter(df2)
    pairg = zip(plot_detections(fdpg1, gdf1), plot_detections(fdpg2, gdf2))
    fdp1, fdp2 = next(pairg)
    frame = np.concatenate((fdp1.image, fdp2.image), axis=1)
    with create_avi(destination, frame) as append_fn:
        for fdp1, fdp2 in pairg:
            frame = np.concatenate((fdp1.image, fdp2.image), axis=1)
            append_fn(frame)


def compare_raw_vs_filtered(video_file):
    model = create_model()
    if not os.path.exists("df_raw.csv"):
        df_raw = get_raw_df_from_movie(video_file, model)
        df_raw.to_csv("df_raw.csv")
    else:
        df_raw = pd.read_csv("df_raw.csv")
    df_fil = get_tracked_df_from_df(df_raw)
    compare_two_dataframes(video_file, df_raw, df_fil, destination=video_file.replace(':', '_').replace('\\', '_'))


def main():
    video_file = r'D:\aau-rainsnow\Hasserisvej\Hasserisvej-2\cam1.mkv'
    compare_raw_vs_filtered(video_file)

if __name__ == "__main__":
    main()