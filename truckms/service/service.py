import os
from werkzeug import secure_filename
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


def image2htmlstr(image):
    is_success, buffer = cv2.imencode(".jpg", image)
    io_buf = io.BytesIO(buffer)
    figdata_png = base64.b64encode(io_buf.getvalue())
    result = str(figdata_png)[2:-1]
    return result


def html_imgs_generator(video_path):
    csv_file_path = os.path.splitext(video_path)[0] + ".csv"
    pred_gen_from_df = pandas_to_pred_iter(pd.read_csv(csv_file_path))
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
    root = Tk()
    root.withdraw()
    # ensure the file dialog pops to the top window
    root.wm_attributes('-topmost', 1)
    fname = askopenfilename(parent=root)
    return fname


def create_microservice(db_url, dispatch_work_func):
    """
    Args:
        db_url: url to a database
        dispatch_work_func: function that receives a string as argument. This argument is a video file path. The function
            should do something with that file
    """
    app = Flask(__name__, template_folder=osp.join(osp.dirname(__file__), 'templates'),
                static_folder=osp.join(osp.dirname(__file__), 'templates', 'assets'))

    bootstrap = Bootstrap(app)

    @app.route("/upload_recordings", methods=['POST'])
    def upload_recordings():
        for filename in request.files:
            f = request.files[filename]
            filename = secure_filename(filename)
            filepath = os.path.join(upload_directory, filename)
            f.save(filepath)
            app.worker_pool.apply_async(func=analyze_movie, args=(filepath, max_operating_res, skip))
            app.logger.info('started this shit')

        return redirect(url_for("index"))

    @app.route('/check_status')
    def check_status():
        from truckms.service.model import create_session, VideoStatuses
        session = create_session(db_url)
        query = VideoStatuses.get_video_statuses(session)
        video_items = []
        for item in query:
            video_items.append({'filename': item.file_path,
                                'status': 'ready' if item.results_path is not None else 'processing'})
        partial_destination_url = '/show_video?filename='
        resp = make_response(render_template("check_status.html", partial_destination_url=partial_destination_url,
                                             video_items=video_items))
        return resp

    @app.route('/file_select')
    def file_select():
        if request.remote_addr != '127.0.0.1':
            make_response("Just what do you think you're doing, Dave?", 403)

        fname = gui_select_file()
        dispatch_work_func(fname)

        return redirect(url_for("check_status"))

    @app.route('/show_video')
    def show_video():
        filepath = osp.join(upload_directory, request.args.get('filename'))
        plots_gen = html_imgs_generator(filepath)

        try:
            first_image = next(plots_gen)
            resp = make_response(render_template("show_video.html", first_image_str=first_image, images=plots_gen))
        except StopIteration:
            resp = make_response(render_template("show_video.html", first_image_str='', images=[]))
        return resp

    @app.route('/')
    def root():
        return redirect(url_for("index"))

    @app.route('/index')
    def index():
        resp = make_response(render_template("index.html"))
        return resp

    return app