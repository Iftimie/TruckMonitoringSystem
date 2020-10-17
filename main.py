import os.path as osp
import os
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
from p2prpc.errors import ClientFutureTimeoutError
import signal
import datetime
from p2prpc import monitoring
from function import p2prpc_analyze_movie


password = "super secret password"
root_dir = '/home/achellaris/projects_data/main_dir'
path = osp.join(root_dir, 'clientdb')
client_app = create_p2p_client_app(osp.join(root_dir, "network_discovery_client.txt"), password=password, cache_path=path)
dec_analyze_movie = client_app.register_p2p_func()(p2prpc_analyze_movie)

app = Flask(__name__, template_folder=osp.join(osp.dirname(__file__), 'templates'),
            static_folder=osp.join(osp.dirname(__file__), 'templates', 'assets'))
Bootstrap(app)
#/home/achellaris/projects/TruckMonitoringSystem/tests/truckms/service/data/cut.mkv
#/home/achellaris/big_data/tms_data/good_data/output_8.mp4


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
        res = dec_analyze_movie(video_handle=open(fname, 'rb'))
        try:
            res.get(5)
        except ClientFutureTimeoutError:
            pass
        return redirect(url_for("check_status"))
    else:
        return redirect(url_for("index"))


@app.route('/check_status')
def check_status():
    video_items = []
    all_items = monitoring.function_call_states(client_app)

    for future in all_items:
        res = None
        status = 'processing'
        tstamp2dtime = datetime.datetime.utcfromtimestamp(future.p2pclientarguments.p2parguments.timestamp)
        try:
            future.get(10)
            status = 'ready'
        except ClientFutureTimeoutError:
            pass
        video_basename = osp.basename(future.p2pclientarguments.p2parguments.kwargs['video_handle'].name)
        progress = future.p2pclientarguments.p2parguments.outputs['progress']
        video_items.append({'filename': video_basename,
                            'status': status,
                            'progress': progress,
                            'time_of_request': tstamp2dtime,
                            'identifier': future.p2pclientarguments.p2parguments.args_identifier})

    partial_destination_url = '/show_video?doc_id='
    restart_url = '/restart_function?doc_id='
    resp = make_response(render_template("check_status.html",
                                         partial_destination_url=partial_destination_url,
                                         restart_url=restart_url,
                                         video_items=video_items))
    return resp


@app.route('/show_video')
def show_video():
    item = monitoring.item_by_func_and_id(client_app, p2prpc_analyze_movie, request.args.get('doc_id'))
    plots_gen = html_imgs_generator(item.p2parguments.outputs['video_results'].name, item.p2parguments.outputs['results'].name)
    try:
        first_image = next(plots_gen)
        video_url = "/cdn" + item.p2parguments.outputs['video_results'].name
        resp = make_response(render_template("show_video.html", first_image_str=first_image, images=plots_gen,
                                             video_results=video_url))
    except StopIteration:
        resp = make_response(render_template("show_video.html", first_image_str='', images=[]))
    return resp


@app.route('/restart_function')
def restart_function():
    future = client_app.create_future(analyze_movie, request.args.get('doc_id'))
    future.restart_func()
    resp = redirect(url_for('check_status'))
    return resp


# Custom static data
@app.route('/cdn/<path:filename>')
def custom_static(filename):
    return send_from_directory("/"+osp.dirname(filename), osp.basename(filename))


@app.route('/')
def root():
    return redirect(url_for("index"))


@app.route('/index')
def index():
    resp = make_response(render_template("index.html"))
    return resp


@app.route("/show_nodestates")
def show_workers():
    res = monitoring.get_node_states("localhost:5000")
    resp = make_response(render_template("show_nodestates.html", worker_list=res))
    return resp

uiport = 3000
uiapp = FlaskUI(app, host="localhost", port=uiport)
uiapp.browser_path = uiapp.get_default_chrome_path()
uiapp.browser_thread = Thread(target=partial(open_browser_func, uiapp, localhost='http://127.0.0.1:{}/'.format(uiport)))

uiapp.run()
