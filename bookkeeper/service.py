from flask import Flask, render_template, make_response, request, redirect, url_for
from werkzeug import secure_filename


def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'


def node_states():
    content = request.json
    state = {"ip": request.environ['REMOTE_ADDR'],
             "port": request.environ['REMOTE_PORT']}
    print (content)
    return make_response("done", 200)

def create_microservice(debug=False):
    app = Flask(__name__)

    if debug:
        app.route('/shutdown', methods=['POST'])(shutdown)
    app.route("/node_states", methods=['POST'])(node_states)



    return app