from flask import Blueprint, request, make_response, Flask
from functools import wraps, partial
from werkzeug import secure_filename
from truckms.service.model import create_session, VideoStatuses, HeartBeats
import requests
import os


def heartbeat(db_url):
    """
    Pottential vulnerability from flooding here
    """
    session = create_session(db_url)
    HeartBeats.add_heartbeat(session)
    session.close()
    return make_response("Thank god you are alive", 200)


def create_broker_blueprint(up_dir, db_url):
    broker_bp = Blueprint("broker_bp", __name__)
    heartbeat_func = (wraps(heartbeat)(partial(heartbeat, db_url)))
    broker_bp.route("/heartbeat", methods=['POST'])(heartbeat_func)
    broker_bp.role = "broker"
    return broker_bp


def create_broker_microservice(up_dir, db_url):
    app = Flask(__name__)
    app.roles = []
    broker_bp = create_broker_blueprint(up_dir, db_url)
    app.register_blueprint(broker_bp)
    app.roles.append(broker_bp.role)
    return app
