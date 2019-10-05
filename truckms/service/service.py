from flask import Flask, request, json
import os
from werkzeug import secure_filename
from flask import Response
import multiprocessing
from truckms.inference.neural import TruckDetector
from truckms.inference.utils import image_generator


def analyze_movie(video_path, max_operating_res):
    p = TruckDetector(max_operating_res=max_operating_res, batch_size=10)
    image_gen = image_generator(video_path, skip=0)
    pred_gen = p.compute(image_gen)
    df = p.pred_iter_to_pandas(pred_gen)
    df.to_csv(os.path.splitext(video_path)[0]+'.csv')


def create_microservice(upload_directory="tms_upload_dir", num_workers=1, max_operating_res=800):
    """
    Creates a microservice ready to run. This microservice will accept upload requests. It has a default
    upload_directory that will be created relative to the current directory.
    """
    app = Flask(__name__)
    if not os.path.exists(upload_directory):
        os.mkdir(upload_directory)

    app.worker_pool = multiprocessing.Pool(num_workers)

    @app.route("/upload_recording", methods=['POST'])
    def upload_weights():

        files_dict = {}
        for filename in request.files:
            f = request.files[filename]
            filename = secure_filename(filename)
            filepath = os.path.join(upload_directory, filename)
            files_dict[filename] = filepath
            f.save(filepath)
            app.worker_pool.apply_async(func=analyze_movie, args=(filepath, max_operating_res))

        return Response(json.dumps({"message": 'files uploaded successfully and started analyzing'}),
                        status=200, mimetype='application/json')
    return app