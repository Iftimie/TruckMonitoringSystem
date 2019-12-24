import requests
from flask import request, jsonify
from json import dumps, loads
from werkzeug import secure_filename
from truckms.service_v2.api import P2PBlueprint
from functools import wraps, partial
from copy import deepcopy
import logging
import io
import traceback
import tinymongo
from truckms.service_v2.api import self_is_reachable
import os
from typing import Tuple
logger = logging.getLogger(__name__)


def default_serialize(data) -> Tuple[dict, str]:
    """
    data should be a dictionary
    return tuple containing a dictionary with handle for a single file and json
    example:
    {}, json string: '{"key":value}'
    {"filename":handle}, '{}'
    """
    files = {os.path.basename(v.name): v for k, v in data.items() if isinstance(v, io.IOBase)}
    files_significance = {os.path.basename(v.name): k for k, v in data.items() if isinstance(v, io.IOBase)}
    new_data = {k: v for k, v in data.items() if not isinstance(v, io.IOBase)}
    new_data["files_significance"] = files_significance
    json_obj = dumps(new_data)
    return files, json_obj


def default_deserialize(files, json, up_dir):
    """
    both are dictionaries
    the files dict should contain a single handle to file
    #TODO implement for .ZIP file
    """
    data = loads(json)

    listkeys = list(files.keys())
    assert len(listkeys) <= 1
    if listkeys:
        filename = listkeys[0]
        filepath = os.path.join(up_dir, filename)
        files[filename].save(filepath)
        sign = data["files_significance"][filename]
        data[sign] = filepath
    del data["files_significance"]

    return data


def p2p_route_insert_one(db_path, db, col, deserializer=default_deserialize, current_address_func=lambda:None):
    if request.files:
        filename = list(request.files.keys())[0]
        files = {secure_filename(filename): request.files[filename]}
    else:
        files = dict()

    data_to_insert = deserializer(files, request.form['json'])
    data_to_insert["current_address"] = current_address_func()
    # collection.insert_one(data_to_insert)
    update_one(db_path, db, col, data_to_insert, data_to_insert, upsert=True)


def p2p_route_update_one(db_path, db, col, deserializer=default_deserialize, post_func=requests.post):
    if request.files:
        filename = list(request.files.keys())[0]
        files = {secure_filename(filename): request.files[filename]}
    else:
        files = dict()
    update_data = deserializer(files, request.form['update_json'])
    filter_data = deserializer({}, request.form['filter_json'])
    visited_nodes = loads(request.form['visited_json'])
    # update_one(db_path, db, col, filter_data, update_data, upsert=False)
    visited_nodes = p2p_update_one(db_path, db, col, filter_data, update_data, post_func=post_func, visited_nodes=visited_nodes)
    return jsonify(visited_nodes)


def create_p2p_blueprint(up_dir, db_url, post_func, current_address_func=lambda: None):
    p2p_blueprint = P2PBlueprint("p2p_blueprint", __name__, role="storage")
    new_deserializer = partial(default_deserialize, up_dir=up_dir)
    p2p_route_insert_one_func = (wraps(p2p_route_insert_one)(partial(p2p_route_insert_one, db_path=db_url, deserializer=new_deserializer, current_address_func=current_address_func)))
    p2p_blueprint.route("/insert_one/<db>/<col>", methods=['POST'])(p2p_route_insert_one_func)
    p2p_route_update_one_func = (wraps(p2p_route_update_one)(partial(p2p_route_update_one, db_path=db_url, deserializer=new_deserializer, post_func=post_func)))
    p2p_blueprint.route("/update_one/<db>/<col>", methods=['POST'])(p2p_route_update_one_func)
    return p2p_blueprint


# def insert_one(db_path, db, col, data):
#
#     collection = tinymongo.TinyMongoClient(db_path)[db][col]
#     data = deepcopy(data)
#     data["nodes"] = []
#     collection.insert(data)

def separate_io_data(data):
    files = {k: v for k, v in data.items() if isinstance(v, io.IOBase)}
    new_data = {k: v for k, v in data.items() if not isinstance(v, io.IOBase)}
    for k in files:
        new_data[k] = files[k].name
    return files, new_data


def update_one(db_path, db, col, query, doc, upsert=False):
    collection = tinymongo.TinyMongoClient(db_path)[db][col]
    _, query = separate_io_data(query)
    _, doc = separate_io_data(doc)
    res = list(collection.find(query))
    if len(res) == 0 and upsert is True:
        if "nodes" not in doc:
            doc["nodes"] = []
        collection.insert(doc)
    elif len(res) == 1:
        collection.update_one(query, {"$set": doc})
    else:
        raise ValueError("Unable to update. Query: {}. Documents: {}".format(str(query), str(res)))


def p2p_insert_one(db_path, db, col, data, nodes, serializer=default_serialize, post_func=requests.post, current_address_func=lambda: None):
    """
    post_func is used especially for testing
    current_address_func: self_is_reachable should be called
    """
    current_addr = current_address_func()
    try:
        update = data
        update["nodes"] = nodes
        update["current_address"] = current_addr
        update_one(db_path, db, col, data, update, upsert=True)
    except ValueError as e:
        logger.info(traceback.format_exc())
        raise e

    for i, node in enumerate(nodes):
        # the data sent to a node will not contain in "nodes" list the pointer to that node. only to other nodes
        data["nodes"] = nodes[:i] + nodes[i+1:]
        data["nodes"] += [current_addr] if current_addr else []
        file, json = serializer(data)
        del data["nodes"]
        try:
            post_func("http://{}/insert_one/{}/{}".format(node, db, col), files=file, json={"json": json})
        except:
            traceback.print_exc()
            logger.info(traceback.format_exc())
            logger.info("Unable to post p2p data")


def p2p_update_one(db_path, db, col, filter, update, serializer=default_serialize, post_func=requests.post, visited_nodes=None):
    if visited_nodes is None:
        visited_nodes = []
    try:
        update_one(db_path, db, col, filter, update, upsert=False)
    except ValueError as e:
        logger.info(traceback.format_exc())
        raise e

    collection = tinymongo.TinyMongoClient(db_path)[db][col]
    res = list(collection.find(filter))
    if len(res) != 1:
        raise ValueError("Unable to update. Query: {}. Documents: {}".format(str(filter), str(res)))

    nodes = res[0]["nodes"]
    current_node = res[0]["current_address"]
    visited_nodes.append(current_node)

    for i, node in enumerate(nodes):
        if node in visited_nodes:
            continue

        files, update_json = serializer(update)
        _, filter_json = serializer(filter)
        visited_json = dumps(visited_nodes)

        try:
            res = post_func("http://{}/update_one/{}/{}".format(node, db, col), files=files, json={"update_json": update_json,
                                                                                                   "filter_json": filter_json,
                                                                                                   "visited_json": visited_json})
            if res.status_code == 200:
                visited_nodes = list(set(visited_nodes) | set(res.json))
        except:
            traceback.print_exc()
            logger.info(traceback.format_exc())
            logger.info("Unable to post p2p data")

    return visited_nodes
