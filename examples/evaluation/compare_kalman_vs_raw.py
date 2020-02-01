from truckms.evaluation.comparison import compare_multiple_dataframes
from truckms.inference.neural import create_model, compute, pred_iter_to_pandas, pandas_to_pred_iter
from truckms.inference.analytics import filter_pred_detections
from truckms.inference.utils import framedatapoint_generator
import os
import pandas as pd
import os.path as osp


def get_raw_df_from_movie(movie_path, model):
    g1 = framedatapoint_generator(movie_path, skip=0, max_frames=200)
    g2 = compute(g1, model, filter_classes=['train', 'truck', 'bus'])
    df = pred_iter_to_pandas(g2)
    return df


def get_tracked_df_from_df(df):
    g1 = filter_pred_detections(pandas_to_pred_iter(df))
    filtered_df = pred_iter_to_pandas(g1)
    return filtered_df


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
    """
    This comparison will compare the results between a pipeline with no filtering (raw detections) and a pipeline with
    kalman filtering and kuhn-munkres(hungarian) algorithm for pair matching.
    """
    # run script download_experimental_data.bat before running this script in order to have the video
    video_file = osp.join(osp.dirname(__file__),'..', '4K Traffic camera video - free download now!-MNn9qKG2UFI.webm')
    compare_raw_vs_filtered(video_file)


if __name__ == "__main__":
    main()
