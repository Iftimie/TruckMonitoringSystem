from truckms.service.worker.server import analyze_movie
from truckms.service.gui_interface import html_imgs_generator
import os.path as osp
import os
import cv2

def progress(idx, end):
    print(idx, end)

def run_every_frame(video_path):
    directory = "results_every_frame"
    if not osp.exists(directory):
        os.mkdir(directory)
    from truckms.service import gui_interface
    gui_interface.image2htmlstr = lambda image: image
    analyze1_csv = analyze_movie(video_path, progress_hook=progress)
    plots_gen = html_imgs_generator(video_path, analyze1_csv)
    for idx, image in enumerate(plots_gen):
        cv2.imwrite(osp.join(directory, "{}.png".format(idx)), image)


if __name__ == "__main__":
    video_path = r'D:\tms_data\concatenated.avi'
    video_path = r'D:\tms_data\good_data\output_0.mp4'
    run_every_frame(video_path)

