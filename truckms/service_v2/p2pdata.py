import requests
from flask import request, jsonify, send_file, make_response
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
import time
from functools import wraps
import inspect


def fixate_args(**fixed_kwargs):
    """
    Unlike functools.partial where you fixate the arguments, maybe we are unable to fixate the arguments because in
    flask we have routing and we cannot ignore the arguments that are passed from routing.
    For example we may fixate some arguments in p2p_route_push_update_one() for security reasons. We may want db and collection
    to be only "mydb" and "mycollection". One way to do this would be to call functools.partial.
    But that function is used in flask routing, and other functions from this p2pframework such as p2p_push_update_one
    will call the function with duplicated arguments.
    """
    def inner_decorator(f):
        formal_args = list(inspect.signature(f).parameters.keys())
        positions_to_check = {i:arg for i, arg in enumerate(formal_args) if arg in fixed_kwargs}
        @wraps(f)
        def wrap(*args, **kwargs):
            for i, val in enumerate(args):
                if i in positions_to_check and not val == fixed_kwargs[positions_to_check[i]]:
                    raise ValueError("bad argument: expected ", fixed_kwargs[positions_to_check[i]], "found", val)
            for k in kwargs:
                if k in fixed_kwargs and not kwargs[k] == fixed_kwargs[k]:
                    raise ValueError("bad argument: expected ", fixed_kwargs[k], "found", kwargs[k])
            return f(*args, **kwargs)
        return wrap
    return inner_decorator


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
        original_filename = listkeys[0]
        filepath = os.path.join(up_dir, original_filename)
        original_filepath = filepath
        i = 1
        while os.path.exists(filepath):
            filename, file_extension = os.path.splitext(original_filepath)
            filepath = filename + "_{}_".format(i) + file_extension
            i += 1
        files[original_filename].save(filepath)
        sign = data["files_significance"][original_filename]
        data[sign] = filepath
    del data["files_significance"]

    return data


def p2p_route_insert_one(db_path, db, col, deserializer=default_deserialize, current_address_func=lambda: None):
    if request.files:
        filename = list(request.files.keys())[0]
        files = {secure_filename(filename): request.files[filename]}
    else:
        files = dict()

    data_to_insert = deserializer(files, request.form['json'])
    data_to_insert["current_address"] = current_address_func()
    # collection.insert_one(data_to_insert)
    update_one(db_path, db, col, data_to_insert, data_to_insert, upsert=True)


# def p2p_route_insert_many(db_path, db, col, deserializer=default_deserialize, current_address_func=lambda: None):
#     if request.files:
#         raise ValueError("Not implemented for many documents with files")
#
#     data_to_insert = deserializer({}, request.form['json'])
#     data_to_insert["current_address"] = current_address_func()
#     # collection.insert_one(data_to_insert)
#     update_one(db_path, db, col, data_to_insert, data_to_insert, upsert=True)


def p2p_route_push_update_one(db_path, db, col, deserializer=default_deserialize):
    if request.files:
        filename = list(request.files.keys())[0]
        files = {secure_filename(filename): request.files[filename]}
    else:
        files = dict()
    update_data = deserializer(files, request.form['update_json'])
    filter_data = deserializer({}, request.form['filter_json'])
    visited_nodes = loads(request.form['visited_json'])
    recursive = request.form["recursive"]
    # update_one(db_path, db, col, filter_data, update_data, upsert=False)
    if recursive:
        visited_nodes = p2p_push_update_one(db_path, db, col, filter_data, update_data, visited_nodes=visited_nodes)
    return jsonify(visited_nodes)


def p2p_route_pull_update_one(db_path, db, col, serializer=default_serialize):
    req_keys = loads(request.form['req_keys_json'])
    filter_data = loads(request.form['filter_json'])
    hint_file_keys = loads(request.form['hint_file_keys_json'])
    required_list = list(tinymongo.TinyMongoClient(db_path)[db][col].find(filter_data))
    assert len(required_list) == 1
    required_data = required_list[0]
    data_to_send = {}
    for k in req_keys:
        if k not in hint_file_keys:
            data_to_send[k] = required_data[k]
        else:
            data_to_send[k] = open(required_data[k])
    files, json = serializer(data_to_send)
    if files:
        k = list(files.keys())[0]
        result = send_file(files[k].name, as_attachment=True)
        result.headers['update_json'] = json
    else:
        result = make_response("")
        # result = send_file(__file__, as_attachment=True)
        result.headers['update_json'] = json

    return result


def create_p2p_blueprint(up_dir, db_url, current_address_func=lambda: None):
    p2p_blueprint = P2PBlueprint("p2p_blueprint", __name__, role="storage")
    new_deserializer = partial(default_deserialize, up_dir=up_dir)
    p2p_route_insert_one_func = (wraps(p2p_route_insert_one)(partial(p2p_route_insert_one, db_path=db_url, deserializer=new_deserializer, current_address_func=current_address_func)))
    p2p_blueprint.route("/insert_one/<db>/<col>", methods=['POST'])(p2p_route_insert_one_func)
    p2p_route_push_update_one_func = (wraps(p2p_route_push_update_one)(partial(p2p_route_push_update_one, db_path=db_url, deserializer=new_deserializer)))
    p2p_blueprint.route("/push_update_one/<db>/<col>", methods=['POST'])(p2p_route_push_update_one_func)
    p2p_route_pull_update_one_func = (wraps(p2p_route_pull_update_one)(partial(p2p_route_pull_update_one, db_path=db_url)))
    p2p_blueprint.route("/pull_update_one/<db>/<col>", methods=['POST'])(p2p_route_pull_update_one_func)
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


def validate_document(document):
    for k in document:
        if isinstance(document[k], dict):
            raise ValueError("Cannot have nested dictionaries in current implementation")


def update_one(db_path, db, col, query, doc, upsert=False):
    """
    This function is entry point for both insert and update.
    """
    validate_document(doc)
    collection = tinymongo.TinyMongoClient(db_path)[db][col]
    _, query = separate_io_data(query)
    _, doc = separate_io_data(doc)
    res = list(collection.find(query))
    doc["timestamp"] = time.time()
    if len(res) == 0 and upsert is True:
        if "nodes" not in doc:
            doc["nodes"] = []
        collection.insert(doc)
    elif len(res) == 1:
        collection.update_one(query, {"$set": doc})
    else:
        raise ValueError("Unable to update. Query: {}. Documents: {}".format(str(query), str(res)))


def p2p_insert_one(db_path, db, col, data, nodes, serializer=default_serialize, current_address_func=lambda: None):
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
            requests.post("http://{}/insert_one/{}/{}".format(node, db, col), files=file, data={"json": json})
        except:
            traceback.print_exc()
            logger.info(traceback.format_exc())
            logger.info("Unable to post p2p data")


def p2p_push_update_one(db_path, db, col, filter, update, serializer=default_serialize, visited_nodes=None, recursive=True):
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
            res = requests.post("http://{}/push_update_one/{}/{}".format(node, db, col), files=files, data={"update_json": update_json,
                                                                                                   "filter_json": filter_json,
                                                                                                   "visited_json": visited_json,
                                                                                                        "recursive": recursive})
            if res.status_code == 200:
                visited_nodes = list(set(visited_nodes) | set(res.json))
        except:
            traceback.print_exc()
            logger.info(traceback.format_exc())
            logger.info("Unable to post p2p data")

    return visited_nodes


class WrapperSave:
    def __init__(self, data):
        self.bytes = data

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            f.write(self.bytes)


def merge_downloaded_data(original_data, merging_data):
    if not merging_data:
        return {}
    update_keys = merging_data[0].keys()
    result_update = {}
    for k in update_keys:
        if "timestamp" in k:continue
        if all(d[k] == original_data[k] for d in merging_data):
            result_update[k] = max(merging_data, key=lambda d: d['timestamp'])[k]
        else:
            result_update[k] = max(merging_data, key=lambda d: d['timestamp'] if d[k] != original_data[k] else 0)[k]

    return result_update


def p2p_pull_update_one(db_path, db, col, filter, req_keys, deserializer, hint_file_keys=None, merging_func=merge_downloaded_data):
    if hint_file_keys is None:
        hint_file_keys = []

    req_keys = list(set(req_keys) | set(["timestamp"]))

    collection = tinymongo.TinyMongoClient(db_path)[db][col]
    collection_res = list(collection.find(filter))
    if len(collection_res) != 1:
        raise ValueError("Unable to update. Query: {}. Documents: {}".format(str(filter), str(collection_res)))

    nodes = collection_res[0]["nodes"]

    files_to_remove_after_download = []
    merging_data = []

    for i, node in enumerate(nodes):

        req_keys_json = dumps(req_keys)
        filter_json = dumps(filter)
        hint_file_keys_json = dumps(hint_file_keys)


        try:
            res = requests.post("http://{}/pull_update_one/{}/{}".format(node, db, col), files={}, data={"req_keys_json": req_keys_json,
                                                                                          "filter_json": filter_json,
                                                                                          "hint_file_keys_json": hint_file_keys_json})

            if res.status_code == 200:
                update_json = res.headers['update_json']
                files = {}
                if 'Content-Disposition' in res.headers:
                    filename = res.headers['Content-Disposition'].split("filename=")[1]
                    files = {filename: WrapperSave(res.data)}
                downloaded_data = deserializer(files, update_json)
                merging_data.append(downloaded_data)
        except:
            traceback.print_exc()
            logger.info(traceback.format_exc())
            logger.info("Unable to post p2p data")

    update = merging_func(collection_res[0], merging_data)

    try:
        update_one(db_path, db, col, filter, update, upsert=False)
    except ValueError as e:
        logger.info(traceback.format_exc())
        raise e