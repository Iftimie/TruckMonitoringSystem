from truckms.service.worker.worker_client import get_available_brokers
import requests
import logging
logger = logging.getLogger(__name__)
import time
import tinymongo
from truckms.service_v2.userclient.userclient import analyze_and_updatedb
from truckms.service_v2.p2pdata import p2p_pull_update_one, default_deserialize
from functools import partial
from truckms.service.worker.server import analyze_movie


def find_response_with_work(local_port, collection, func_name):

    work_found = False
    res_broker_ip = None
    res_broker_port = None
    res_json = dict()
    while work_found is False:
        brokers = get_available_brokers(local_port=local_port)
        for broker in brokers:
            broker_ip, broker_port = broker['ip'], broker['port']
            try:
                res = requests.get('http://{}:{}/search_work/{}/{}'.format(broker_ip, broker_port, collection, func_name))
                if res.json and 'identifier' in res.json:
                    logger.info("Found work from {}, {}".format(broker_ip, broker_port))
                    work_found = True
                    res_broker_ip = broker_ip
                    res_broker_port = broker_port
                    res_json = res.json
                    break
            except:  # except connection timeout or something like that
                pass
        if work_found is False:
            logger.info("No work found")
        time.sleep(1)

    # TODO it may be possible that res allready contains broker ip and port?
    return res_json, res_broker_ip, res_broker_port


def do_work(up_dir, db_url, local_port, func_registry=None, collection="movie_statuses"):
    if func_registry is None:
        func_registry = {"analyze_and_updatedb": partial(analyze_and_updatedb, db_url=db_url, analysis_func=analyze_movie)}

    frk = list(func_registry.keys())
    assert len(frk) == 1
    func_name = frk[0]
    func = func_registry[func_name]

    res, broker_ip, broker_port = find_response_with_work(local_port, collection, func_name)
    tinymongo.TinyMongoClient(db_url)["tms"][collection].insert_one({"identifier": res['identifier'], "nodes": [broker_ip+":"+str(broker_port)]})

    deserializer = partial(default_deserialize, up_dir=up_dir)
    # TODO instead of hardcoding here the required keys. those keys could be inspected form the function declaration,
    #  or a dectoator should be used to help the framework to know which resources the funciton needs in order to be executed on worker and workerclient.
    #  also the return value of the analysis func should be a dictionary
    #  actually the question is which function should do the pull and which function should do the push
    #  I believe that the user that implements the analysis function should have no responsability of knowing about p2p_data
    #  I will curently keep it like that, but needs refactoring
    p2p_pull_update_one(db_url, "tms", collection, {"identifier": res["identififier"]}, ["video_path"], deserializer, hint_file_keys=["video_path"])

    func(db_url)
