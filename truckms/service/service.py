from flask import Flask, request, json
import os
from werkzeug import secure_filename
from flask import Response
import multiprocessing
from truckms.inference.neural import create_model, pandas_to_pred_iter, pred_iter_to_pandas, plot_detections, compute
from truckms.inference.neural import create_model_efficient
from truckms.inference.utils import framedatapoint_generator, framedatapoint_generator_by_frame_ids2
from flask import Flask, render_template, send_from_directory, make_response, request, redirect, url_for, session, jsonify
import os.path as osp
from flask_bootstrap import Bootstrap
from functools import partial
import base64
import cv2
import io
import pandas as pd
from truckms.inference.analytics import filter_pred_detections, get_important_frames


def analyze_movie(video_path, max_operating_res, skip=0):
    """
    Attention!!! if the movie is short or too fast and skip  is too big, then it may result with no detections
    #TODO think about this
    """
    model = create_model_efficient(model_creation_func=partial(create_model, max_operating_res=max_operating_res))
    image_gen = framedatapoint_generator(video_path, skip=skip)
    pred_gen = compute(image_gen, model=model, batch_size=5)
    filtered_pred = filter_pred_detections(pred_gen)
    df = pred_iter_to_pandas(filtered_pred)
    df.to_csv(os.path.splitext(video_path)[0]+'.csv')


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


def create_microservice(upload_directory="tms_upload_dir", num_workers=1, max_operating_res=800, skip=0):
    """
    Creates a microservice ready to run. This microservice will accept upload requests. It has a default
    upload_directory that will be created relative to the current directory.
    """
    app = Flask(__name__, template_folder=osp.join(osp.dirname(__file__), 'templates'),
                static_folder=osp.join(osp.dirname(__file__), 'templates', 'assets'))

    bootstrap = Bootstrap(app)

    if not os.path.exists(upload_directory):
        os.mkdir(upload_directory)

    app.worker_pool = multiprocessing.Pool(num_workers)

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

    @app.route("/upload_menu")
    def upload_menu():
        resp = make_response(render_template("upload.html"))
        return resp

    @app.route('/check_status')
    def check_status():
        video_items = []
        for file in filter(lambda x: '.csv' not in x, os.listdir(upload_directory)):

            video_items.append({'filename': file,
                                'status': 'ready' if osp.exists(osp.join(upload_directory, os.path.splitext(file)[0]+'.csv')) else 'processing'})
        partial_destination_url = '/show_video?filename='
        resp = make_response(render_template("check_status.html", partial_destination_url=partial_destination_url,
                                             video_items=video_items))
        return resp

    @app.route('/file_select')
    def file_select():
        if request.remote_addr != '127.0.0.1':
            make_response("Just what do you think you're doing, Dave?", 403)
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        root = Tk()
        root.withdraw()
        # ensure the file dialog pops to the top window
        root.wm_attributes('-topmost', 1)
        fname = askopenfilename(parent=root)

        # TODO do something with this filename
        # the file must not be move, however the .csv file must stay somewhere else? nope. it should stay near the file
        # however once the file was registered for working, when check status is reached, it should look for .csv  wherever it may be found
        # app.worker_pool.apply_async(func=analyze_movie, args=(fname, max_operating_res, skip))
        # app.logger.info('started this shit')

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