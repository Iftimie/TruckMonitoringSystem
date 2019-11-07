from flask import Flask, make_response, request, jsonify
from collections import namedtuple
from flask import Blueprint
from functools import partial, wraps


NodeState = namedtuple('NodeState', ['ip', 'port', 'workload', 'hardware', 'nickname', 'node_type', 'email'])


def node_states(set_states):
    if request.method == 'POST':
        received_states = request.json
        received_states = set(NodeState(**content) for content in received_states)
        set_states.update(received_states)
        return make_response("done", 200)
    else:
        return jsonify(list(set_states))


def create_blueprint():
    bookkeeper_bp = Blueprint("bookkeeper_bp", __name__)
    set_states = set()  # TODO replace with a local syncronized database
    func = (wraps(node_states)(partial(node_states, set_states)))
    bookkeeper_bp.route("/node_states", methods=['POST', 'GET'])(func)
    return bookkeeper_bp


def create_microservice():
    app = Flask(__name__)
    app.register_blueprint(create_blueprint())

    return app
