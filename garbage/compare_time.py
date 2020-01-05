from truckms.service.worker.server import analyze_movie
from truckms.service.gui_interface import html_imgs_generator
from truckms.inference.utils import get_video_file_size
from garbage.evaluate_algorithm import compare_multiple_dataframes
import os.path as osp
import os
import cv2
from truckms.inference.motion_map import movement_frames_indexes
import time
from shutil import copyfile
import pandas as pd


def progress(idx, end):
    print(idx, end)


def run_every_frame(video_path):
    directory = "results_every_frame"
    if not osp.exists(directory):
        os.mkdir(directory)
    extension = osp.splitext(video_path)[1]
    destination_csv = osp.join(directory, osp.basename(video_path).replace(extension, "every_frame.csv"))
    if osp.exists(destination_csv):
        return destination_csv
    from truckms.service import gui_interface
    gui_interface.image2htmlstr = lambda image: image
    analyze1_csv = analyze_movie(video_path, progress_hook=progress)
    copyfile(analyze1_csv, destination_csv)

    plots_gen = html_imgs_generator(video_path, analyze1_csv)
    for idx, image in enumerate(plots_gen):
        cv2.imwrite(osp.join(directory, "{}.png".format(idx)), image)

    return destination_csv


def run_by_motion_map(video_path):
    directory = "results_motion_map"
    if not osp.exists(directory):
        os.mkdir(directory)

    extension = osp.splitext(video_path)[1]
    destination_csv = osp.join(directory, osp.basename(video_path).replace(extension, "motion_map.csv"))
    if osp.exists(destination_csv):
        return destination_csv

    start = time.time()
    from truckms.service import gui_interface
    gui_interface.image2htmlstr = lambda image: image
    analyze2_csv = analyze_movie(video_path, progress_hook=progress, select_frame_inds_func=movement_frames_indexes)
    copyfile(analyze2_csv, destination_csv)

    plots_gen = html_imgs_generator(video_path, analyze2_csv)
    for idx, image in enumerate(plots_gen):
        cv2.imwrite(osp.join(directory, "{}.png".format(idx)), image)
    end = time.time()
    print(end-start)
    print(get_video_file_size(video_path))

    return destination_csv


if __name__ == "__main__":
    video_path = r'D:\tms_data\concatenated.avi'
    video_path = r'D:\tms_data\good_data\output_0.mp4'
    every_frame_csv = run_every_frame(video_path)
    motion_csv = run_by_motion_map(video_path)
    compare_multiple_dataframes(video_path, "here.avi", pd.read_csv(motion_csv), pd.read_csv(every_frame_csv))
