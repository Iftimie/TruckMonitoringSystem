from truckms.service.gui_interface import open_browser_func, image2htmlstr, html_imgs_generator, ignore_remote_addresses
from truckms.service.gui_interface import gui_select_file
from truckms.service_v2.api import P2PFlaskApp
from typing import Tuple
from flaskwebgui import FlaskUI  # get the FlaskUI class
from flask_bootstrap import Bootstrap
import os.path as osp
from flask import redirect, url_for, make_response, render_template
from truckms.service.worker.user_client import get_available_nodes
from threading import Thread
from functools import partial


def create_guiservice(db_url: str, dispatch_work_func: callable, port: int) -> Tuple[FlaskUI, P2PFlaskApp]:
    """
    This factory method creates the FlaskUI and P2PFlaskApp. In order to start the application, the method run of
    FlaskUI object must be called. The P2PFlaskApp object can be used to add additional blueprints.

    Args:
        db_url: url to a database
        dispatch_work_func: function that receives a string as argument. This argument is a video file path. The function
            should do something with that file
        port: port for the GUI service

    Returns:
        FlaskUI object
        P2PFlaskApp object
    """

    app = P2PFlaskApp(__name__, template_folder=osp.join(osp.dirname(__file__), '..','..', 'service', 'templates'),
                      static_folder=osp.join(osp.dirname(__file__), '..', '..', 'service', 'templates', 'assets'))

    Bootstrap(app)

    @app.route('/file_select')
    @ignore_remote_addresses
    def file_select():
        fname = gui_select_file()
        if fname != '':
            dispatch_work_func(fname)
            return redirect(url_for("check_status"))
        else:
            return redirect(url_for("index"))

    @app.route('/')
    @ignore_remote_addresses
    def root():
        return redirect(url_for("index"))

    @app.route('/index')
    @ignore_remote_addresses
    def index():
        resp = make_response(render_template("index.html"))
        return resp

    @app.route("/show_nodestates")
    @ignore_remote_addresses
    def show_workers():
        node_list = get_available_nodes(local_port=port)
        resp = make_response(render_template("show_nodestates.html", worker_list=node_list))
        return resp

    uiapp = FlaskUI(app, host="0.0.0.0", port=port)
    uiapp.browser_thread = Thread(
        target=partial(open_browser_func, uiapp, localhost='http://127.0.0.1:{}/'.format(port)))

    return uiapp, app
