from flask import Flask, make_response, request, jsonify
from collections import namedtuple

def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'


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

def create_microservice(debug=False):
    app = Flask(__name__)

    if debug:
        app.route('/shutdown', methods=['POST'])(shutdown)
    app.route("/node_states", methods=['POST', 'GET'])(node_states)



    return app