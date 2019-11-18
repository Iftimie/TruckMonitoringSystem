from truckms.inference.neural import pandas_to_pred_iter, pred_iter_to_pandas, plot_detections
from truckms.inference.utils import framedatapoint_generator_by_frame_ids2
from flask import Flask, render_template, make_response, request, redirect, url_for
import os.path as osp
from flask_bootstrap import Bootstrap
import base64
import cv2
import io
import pandas as pd
from truckms.inference.analytics import filter_pred_detections, get_important_frames
from truckms.service.model import create_session, VideoStatuses
from flask import make_response, request
from functools import wraps
import sys, subprocess as sps
from flaskwebgui import FlaskUI  # get the FlaskUI class
from threading import Thread
from functools import partial


def open_browser_func(self, localhost):
    """
        Adapted from https://github.com/ClimenteA/flaskwebgui/blob/master/src/flaskwebgui.py
    """

    browser_path = self.find_browser()

    if browser_path:
        try:
            if self.app_mode:
                if self.fullscreen:
                    sps.Popen([browser_path, "--start-fullscreen", '--app={}'.format(localhost)],
                              stdout=sps.PIPE, stderr=sps.PIPE, stdin=sps.PIPE)
                else:
                    sps.Popen([browser_path, "--window-size={},{}".format(self.width, self.height),
                               '--app={}'.format(localhost)],
                              stdout=sps.PIPE, stderr=sps.PIPE, stdin=sps.PIPE)
            else:
                sps.Popen([browser_path, localhost],
                          stdout=sps.PIPE, stderr=sps.PIPE, stdin=sps.PIPE)

        except:
            sps.Popen([browser_path, localhost],
                      stdout=sps.PIPE, stderr=sps.PIPE, stdin=sps.PIPE)
    else:
        import webbrowser
        webbrowser.open_new(localhost)


def image2htmlstr(image):
    is_success, buffer = cv2.imencode(".jpg", image)
    io_buf = io.BytesIO(buffer)
    figdata_png = base64.b64encode(io_buf.getvalue())
    result = str(figdata_png)[2:-1]
    return result


def html_imgs_generator(video_path, csv_path):
    pred_gen_from_df = pandas_to_pred_iter(pd.read_csv(csv_path))
    filtered_pred = filter_pred_detections(pred_gen_from_df)
    filtered_dataframe = pred_iter_to_pandas(filtered_pred)

    important_frames, important_df = get_important_frames(filtered_dataframe)
    image_gen = framedatapoint_generator_by_frame_ids2(video_path, important_frames)

    pred_from_important = pandas_to_pred_iter(important_df)

    for fdp in plot_detections(image_gen, pred_from_important):
        yield image2htmlstr(fdp.image)


def ignore_remote_addresses(f):
    @wraps(f)
    def new_f(*args, **kwargs):
        if request.remote_addr != '127.0.0.1':
            make_response("Just what do you think you're doing, Dave?", 403)
        return f(*args, **kwargs)
    return new_f


def gui_select_file():
    """
    Opens a file selection dialog. Returns a string with the filepath
    """
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    root = Tk()
    root.withdraw()
    # ensure the file dialog pops to the top window
    root.wm_attributes('-topmost', 1)
    fname = askopenfilename(parent=root)
    return fname


def create_guiservice(db_url, dispatch_work_func, port):
    """
    Args:
        db_url: url to a database
        dispatch_work_func: function that receives a string as argument. This argument is a video file path. The function
            should do something with that file
    """
    app = Flask(__name__, template_folder=osp.join(osp.dirname(__file__), 'templates'),
                static_folder=osp.join(osp.dirname(__file__), 'templates', 'assets'))

    Bootstrap(app)

    @app.route('/check_status')
    @ignore_remote_addresses
    def check_status():
        session = create_session(db_url)
        VideoStatuses.check_and_download(session)
        VideoStatuses.remove_dead_requests(session)
        # TODO call remove_dead_requests or insteand of removing, just restart them
        query = VideoStatuses.get_video_statuses(session)
        session.close()
        video_items = []

        #TODO also render on HTML the time of execution if it exists

        for item in query:
            video_items.append({'filename': item.file_path,
                                'status': 'ready' if item.results_path is not None else 'processing',
                                'time_of_request': item.time_of_request.strftime(
                                    "%m/%d/%Y, %H:%M:%S") if item.time_of_request is not None else 'none'})
        partial_destination_url = '/show_video?filename='
        resp = make_response(render_template("check_status.html", partial_destination_url=partial_destination_url,
                                             video_items=video_items))
        return resp

    @app.route('/file_select')
    @ignore_remote_addresses
    def file_select():
        fname = gui_select_file()
        if fname != '':
            dispatch_work_func(fname)
            return redirect(url_for("check_status"))
        else:
            return redirect(url_for("index"))

    @app.route('/show_video')
    @ignore_remote_addresses
    def show_video():
        session = create_session(db_url)
        item = VideoStatuses.find_results_path(session, request.args.get('filename'))
        session.close()

        plots_gen = html_imgs_generator(item.file_path, item.results_path)

        try:
            first_image = next(plots_gen)
            resp = make_response(render_template("show_video.html", first_image_str=first_image, images=plots_gen))
        except StopIteration:
            resp = make_response(render_template("show_video.html", first_image_str='', images=[]))
        return resp

    @app.route('/')
    @ignore_remote_addresses
    def root():
        return redirect(url_for("index"))

    @app.route('/index')
    @ignore_remote_addresses
    def index():
        resp = make_response(render_template("index.html"))
        return resp

    uiapp = FlaskUI(app, host="0.0.0.0", port=port)
    uiapp.browser_thread = Thread(target=partial(open_browser_func, uiapp, localhost='http://127.0.0.1:{}/'.format(port)))

    return uiapp, app
