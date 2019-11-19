from truckms.service.model import create_session, VideoStatuses
from truckms.service.worker.server import analyze_and_updatedb, analyze_movie
from functools import partial
import requests
import time
import logging
import traceback
logger = logging.getLogger(__name__)
import os

def get_available_brokers(local_port):
    res1 = []
    try:
        res1 = requests.get('http://localhost:{}/node_states'.format(local_port)).json()  # will get the data defined above
        res1 = [item for item in res1 if 'broker' in item['node_type']]
        if len(res1) == 0:
            logger.info("No broker available")
            return res1
        res1 = sorted(res1, key=lambda x: x['workload'])
    except:
        logger.info(traceback.format_exc())

    # workload for client
    #   client will send file to brokers with workers waiting
    # workload for worker
    #   worker will fetch from brokers with most work to do (i.e. workload 0)
    # a client will not be able to update the the workload of a broker!!!!!
    return res1


def find_response_with_work(local_port):

    work_found = False
    res = None
    res_broker_ip = None
    res_broker_port = None
    while work_found is False:
        brokers = get_available_brokers(local_port=local_port)
        for broker in brokers:
            broker_ip, broker_port = broker['ip'], broker['port']
            try:
                res = requests.get('http://{}:{}/download_recordings'.format(broker_ip, broker_port))
                if res.content != b'Sorry, got no work to do':
                    work_found = True
                    res_broker_ip = broker_ip
                    res_broker_port = broker_port
                    break
            except:  # except connection timeout or something like that
                pass
        time.sleep(1)

    # TODO it may be possible that res allready contains broker ip and port?
    return res, res_broker_ip, res_broker_port


def upload_results(results_path, broker_ip, broker_port):
    # TODO what if broker is no longer reachable?

    files = {os.path.basename(results_path): open(results_path, 'rb')}
    for i in range(10):
        try:
            res = requests.post('http://{}:{}/upload_results'.format(broker_ip, broker_port), files=files, timeout=30)
            if res.content == b"Thank you for your work":
                break
            elif res.content == b"There is no video file for this result":
                # It should be unlikely to happen
                # TODO maybe the brokers should implement a list of files that are in processing mode
                #  just in order to not lose the work done and send back to the corresponding broker
                logger.info(res.content.decode("utf-8"))
                break
                pass
        except:
            logger.info(traceback.format_exc())
            pass

def save_response(up_dir, res):

    filename = res.headers.get('filename')
    max_operating_res = int(res.headers.get('max_operating_res'))
    skip = int(res.headers.get('skip'))
    filepath = os.path.join(up_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(res.content)
    return filepath, max_operating_res, skip


def do_work(up_dir, db_url, local_port):
    res, broker_ip, broker_port = find_response_with_work(local_port=local_port)
    filepath, max_operating_res, skip = save_response(up_dir, res)
    # TODO I should get from somewhere from the response the max_operating_res and skip
    analysis_func = partial(analyze_movie, max_operating_res=max_operating_res, skip=skip)
    results_path = analyze_and_updatedb(db_url, filepath, analysis_func)
    upload_results(results_path, broker_ip, broker_port)

    # for a broker, the workload will be 0 or 100. a worker can signal a broker. when signalling, the workload will be automatically set to 0
    # when a file is sent for processing the workload will be set to 100

    # TODO broker will receive files. the files accumulate. the initial status of the file will be available for processing
    # when a worker will request a file, the files will decumulate and the respective file will have the status of processing
    # the broker will have a background process that will chef from time to time the status of the file.
    # if a file is too old, then it will change it's status back to available for processing
    # when a worker will send back the request, the status of the file will be done
    # when the client checks for a file, it will chef it it is done.

    # there are 3 statuses of a file: available for processing, processing and done
    # a response can only come from the same registered IP and port. if the .csv comes from another ip and port, it is discarded for security reasons
    # workload here will signify
