from flask import Flask, make_response, request, jsonify
from collections import namedtuple
from flask import Blueprint


NodeState = namedtuple('NodeState', ['ip', 'port', 'workload', 'hardware', 'nickname', 'node_type', 'email'])
set_states = set()  # TODO replace with a local syncronized database


def node_states():
    if request.method == 'POST':
        content = request.json
        content['ip'] = request.environ['REMOTE_ADDR']
        set_states.add(NodeState(**content))
        return make_response("done", 200)
    else:
        return jsonify(list(set_states))


def create_blueprint():
    bookkeeper_bp = Blueprint("bookkeeper_bp", __name__)
    bookkeeper_bp.route("/node_states", methods=['POST', 'GET'])(node_states)
    return bookkeeper_bp


def create_microservice():
    app = Flask(__name__)
    app.register_blueprint(create_blueprint())

    return app
