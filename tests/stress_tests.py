from truckms.service_v2.userclient.p2p_client import create_p2p_client_app, ServerThread
from truckms.service_v2.brokerworker.p2p_brokerworker import P2PBrokerworkerApp
import os
import io
from shutil import rmtree
from concurrent.futures import ThreadPoolExecutor
import time


def large_file_function(video_handle: io.IOBase, random_arg: int) -> {"results": io.IOBase}:
    video_handle.close()
    return {"results": open(video_handle.name, 'rb')}


def multiple_client_calls(tmpdir):

    ndclient_path = os.path.join(tmpdir, "ndclient.txt")
    ndcw_path = os.path.join(tmpdir, "ndcw.txt")
    cache_client_dir = os.path.join(tmpdir, "client")
    cache_bw_dir = os.path.join(tmpdir, "bw")
    with open(ndclient_path, "w") as f: f.write("localhost:5001")
    with open(ndcw_path, "w") as f: f.write("localhost:5001")
    client_app = create_p2p_client_app(ndclient_path, local_port=5000, mongod_port=5100, cache_path=cache_client_dir)
    client_large_file_function = client_app.register_p2p_func(can_do_locally_func=lambda :False)(large_file_function)
    broker_worker_app = P2PBrokerworkerApp(None, local_port=5001, mongod_port=5101, cache_path=cache_bw_dir)
    broker_worker_app.register_p2p_func(can_do_locally_func=lambda :True)(large_file_function)
    broker_worker_thread = ServerThread(broker_worker_app)
    broker_worker_thread.start()

    list_futures_of_futures = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for i in range(20):
            future = executor.submit(client_large_file_function, video_handle=open(__file__, 'rb'), random_arg=i)
            list_futures_of_futures.append(future)
        for i in range(20):
            future = executor.submit(client_large_file_function, video_handle=open(__file__, 'rb'), random_arg=i)
            list_futures_of_futures.append(future)

    list_futures = [f.result() for f in list_futures_of_futures]
    assert len(list_futures) == 40
    list_results = [f.get() for f in list_futures]
    assert len(list_results) == 40 and all(isinstance(r, dict) for r in list_results)
    print(os.listdir(cache_bw_dir+"/p2p/large_file_function"))

    client_app.background_server.shutdown()
    broker_worker_thread.shutdown()


def clean_and_create():
    test_dir = "/home/achellaris/delete_test_dir"
    if os.path.exists(test_dir):
        rmtree(test_dir)
        while os.path.exists(test_dir): pass
    os.mkdir(test_dir)
    return test_dir


if __name__ == "__main__":

    multiple_client_calls(clean_and_create())