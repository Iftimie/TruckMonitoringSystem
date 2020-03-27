from examples.function import analyze_movie
import os.path as osp
from truckms.inference.neural import pandas_to_pred_iter, pred_iter_to_pandas, plot_detections
from truckms.inference.utils import framedatapoint_generator_by_frame_ids2
from flask import render_template, redirect, url_for, Flask, request, send_file, send_from_directory
from flask_bootstrap import Bootstrap
import base64
import cv2
import io
import pandas as pd
from truckms.inference.analytics import filter_pred_detections, get_important_frames
from flask import make_response
import subprocess as sps
from flaskwebgui import FlaskUI  # get the FlaskUI class
from threading import Thread
from functools import partial
from p2prpc.p2p_client import create_p2p_client_app
import signal
from p2prpc import monitoring
import datetime


password = "super secret password"
root_dir = '/home/achellaris/projects_data/main_dir'
path = osp.join(root_dir, 'clientdb')
client_app = create_p2p_client_app(osp.join(root_dir, "network_discovery_client.txt"), password=password, cache_path=path)
dec_analyze_movie = client_app.register_p2p_func(can_do_locally_func=lambda: True)(analyze_movie)
app = Flask(__name__, template_folder=osp.join(osp.dirname(__file__), 'templates'),
            static_folder=osp.join(osp.dirname(__file__), 'templates', 'assets'))
Bootstrap(app)
#/home/achellaris/projects/TruckMonitoringSystem/tests/truckms/service/data/cut.mkv


def shutdown_clientapp():
    client_app.background_server.shutdown()


signal.signal(signal.SIGINT, shutdown_clientapp)
signal.signal(signal.SIGTERM, shutdown_clientapp)


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

        except sps.SubprocessError:
            sps.Popen([browser_path, localhost],
                      stdout=sps.PIPE, stderr=sps.PIPE, stdin=sps.PIPE)
    else:
        import webbrowser
        webbrowser.open_new(localhost)


def image2htmlstr(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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


def gui_select_file():
    """
    Opens a file selection dialog. Returns a string with the filepath
    """
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    tkroot = Tk()
    tkroot.withdraw()
    # ensure the file dialog pops to the top window
    tkroot.wm_attributes('-topmost', 1)
    fname = askopenfilename(parent=tkroot)
    return fname


@app.route('/file_select')
def file_select():
    fname = gui_select_file()
    if fname != '':
        dec_analyze_movie(video_handle=open(fname, 'rb'))
        return redirect(url_for("check_status"))
    else:
        return redirect(url_for("index"))


@app.route('/check_status')
def check_status():
    video_items = []
    all_items = monitoring.function_call_states(client_app)

    for item in all_items:
        tstamp2dtime = 'none'
        status = 'processing'
        if item['timestamp']:
            tstamp2dtime = datetime.datetime.utcfromtimestamp(item['timestamp'])
        if item['video_results']:
            status = 'ready'
        video_items.append({'filename': item['video_handle'].name,
                            'status': status,
                            'progress': item['progress'],
                            'time_of_request': tstamp2dtime,
                            'identifier': str(item['identifier'])},
                           )
    partial_destination_url = '/show_video?doc_id='
    resp = make_response(render_template("check_status.html", partial_destination_url=partial_destination_url,
                                         video_items=video_items))
    return resp


@app.route('/show_video')
def show_video():
    item = monitoring.item_by_func_and_id(client_app, analyze_movie, request.args.get('doc_id'))

    plots_gen = html_imgs_generator(item['video_handle'].name, item['results'].name)

    try:
        first_image = next(plots_gen)
        video_url = "/cdn" + item['video_results'].name
        print(video_url)
        resp = make_response(render_template("show_video.html", first_image_str=first_image, images=plots_gen,
                                             video_results=video_url))
    except StopIteration:
        resp = make_response(render_template("show_video.html", first_image_str='', images=[]))
    return resp


# Custom static data
@app.route('/cdn/<path:filename>')
def custom_static(filename):
    print("heereee", filename)
    print("/"+osp.dirname(filename), osp.basename(filename))
    return send_from_directory("/"+osp.dirname(filename), osp.basename(filename))

@app.route('/')
def root():
    return redirect(url_for("index"))


@app.route('/index')
def index():
    resp = make_response(render_template("index.html"))
    return resp


uiapp = FlaskUI(app, host="0.0.0.0", port=5001)
uiapp.browser_thread = Thread(target=partial(open_browser_func, uiapp, localhost='http://127.0.0.1:{}/'.format(5001)))

uiapp.run()
