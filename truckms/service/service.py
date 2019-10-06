from flask import Flask, request, json
import os
from werkzeug import secure_filename
from flask import Response
import multiprocessing
from truckms.inference.neural import TruckDetector
from truckms.inference.utils import image_generator, image_generator_by_frame_ids
from flask import Flask, render_template, send_from_directory, make_response, request, redirect, url_for, session
import os.path as osp
from flask_bootstrap import Bootstrap
import base64
import cv2
import io
import pandas as pd
from truckms.inference.analytics import filter_pred_detections, get_important_frames


def analyze_movie(video_path, max_operating_res):
    p = TruckDetector(max_operating_res=max_operating_res, batch_size=10)
    image_gen = image_generator(video_path, skip=0)
    pred_gen = p.compute(image_gen)
    filtered_pred = filter_pred_detections(pred_gen)
    df = p.pred_iter_to_pandas(filtered_pred)
    df.to_csv(os.path.splitext(video_path)[0]+'.csv')


def image2htmlstr(image):
    is_success, buffer = cv2.imencode(".jpg", image)
    io_buf = io.BytesIO(buffer)
    figdata_png = base64.b64encode(io_buf.getvalue())
    result = str(figdata_png)[2:-1]
    return result


def html_imgs_generator(video_path):
    csv_file_path = os.path.splitext(video_path)[0] + ".csv"
    pred_gen_from_df = TruckDetector.pandas_to_pred_iter(pd.read_csv(csv_file_path))
    filtered_pred = filter_pred_detections(pred_gen_from_df)
    filtered_dataframe = TruckDetector.pred_iter_to_pandas(filtered_pred)

    important_frames, important_df = get_important_frames(filtered_dataframe)
    image_gen = image_generator_by_frame_ids(video_path, important_frames)

    pred_from_important = TruckDetector.pandas_to_pred_iter(important_df)

    for image, id_ in TruckDetector.plot_detections(image_gen, pred_from_important):
        print (id_)
        yield image2htmlstr(image)


def create_microservice(upload_directory="tms_upload_dir", num_workers=1, max_operating_res=800):
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
            app.worker_pool.apply_async(func=analyze_movie, args=(filepath, max_operating_res))
            app.logger.info('started this shit')

        return redirect(url_for("index"))


    @app.route("/upload_menu")
    def upload_menu():
        resp = make_response(render_template("upload.html"))
        return resp

    @app.route('/check_status')
    def check_status_menu():
        video_items = []
        for file in filter(lambda x: '.csv' not in x, os.listdir(upload_directory)):

            video_items.append({'filename': file,
                                'status': 'ready' if osp.exists(osp.join(upload_directory, os.path.splitext(file)[0]+'.csv')) else 'processing'})
        partial_destination_url = '/show_video?filename='
        resp = make_response(render_template("check_status.html", partial_destination_url=partial_destination_url,
                                             video_items=video_items))
        return resp


    @app.route('/show_video')
    def show_video():
        filepath = osp.join(upload_directory, request.args.get('filename'))
        plots_gen = html_imgs_generator(filepath)

        resp = make_response(render_template("show_video.html", first_image_str=next(plots_gen), images=plots_gen))
        return resp


    @app.route('/')
    def root():
        return redirect(url_for("index"))

    @app.route('/index')
    def index():
        resp = make_response(render_template("index.html"))
        return resp


    return app