import montydb
import requests
from flask import request, jsonify
from werkzeug import secure_filename
from flask import Blueprint
from functools import wraps, partial
from copy import deepcopy
import logging
import io
logger = logging.getLogger(__name__)


def default_serialize(data):
    """
    data should be a dictionary
    return tuple containing a dictionary with handle for a single file and json
    example:
    {}, {"key":value}
    {"filename":handle}, {}
    """
    files = {k: v for k, v in data.items() if isinstance(v, io.IOBase)}
    if files and "files_significance" not in data:
        raise ValueError("key files_significance must be in data")
    for k in files:
        del data[k]
    json = jsonify(data)
    return files, json


def default_deserialize(file, json):
    """
    both are dictionaries
    #TODO implement for .ZIP file
    """
    listkeys = list(file.keys())
    assert len(listkeys) <= 1
    if listkeys:
        filename = listkeys[0]
        filepath = filename  # os.path.join(up_dir, filename)
        file[filepath].save(filepath)
        sign = json["files_significance"][filename]
        json[sign] = filepath
        del json["files_significance"]

    return json

def p2p_route_insert_one(db_path, db, col, deserializer=default_deserialize):
    collection = montydb.MontyClient(db_path)[db][col]

    if request.files:
        filename = request.files[0]
        file = {secure_filename(filename), request.files[filename]}
    else:
        file = {}
    json = request.json

    data_to_insert = deserializer(file, json)
    # what do I do with the file?
    data_to_insert["nodes"].append(request.remote_url)
    collection.insert_one(data_to_insert)



def create_p2p_blueprint(db_url):
    p2p_blueprint = Blueprint("p2p_blueprint", __name__)
    p2p_route_insert_one_func = (wraps(p2p_route_insert_one)(partial(p2p_route_insert_one, db_url)))
    p2p_blueprint.route("/insert_one/<db>/<col>", methods=['POST'])(p2p_route_insert_one_func)


def insert_one(db_path, db, col, data):
    collection = montydb.MontyClient(db_path)[db][col]
    data = deepcopy(data)
    data["nodes"] = []
    collection.insert_one(data)


def p2p_insert_one(db_path, db, col, data, nodes, serializer=default_serialize, post_func=requests.post):
    """
    post_func is used especially for testing
    """
    collection = montydb.MontyClient(db_path)[db][col]
    update = deepcopy(data)
    update["nodes"] = nodes
    collection.update_one(data, update)
    for i, node in enumerate(nodes):
        data_to_send = deepcopy(data)
        data_to_send["nodes"] = nodes[:i] + nodes[i+1:]
        file, json = serializer(data_to_send)
        try:
            post_func("http://{}/insert_one/{}/{}".format(node, db, col), files=file, json=json)
        except:
            logger.info("Unable to post p2p data")


def p2p_update(db, col, filter, update):
    collection = montydb.MontyClient[db][col]
    item = collection.find(filter)
    collection.update_one(filter, update)


