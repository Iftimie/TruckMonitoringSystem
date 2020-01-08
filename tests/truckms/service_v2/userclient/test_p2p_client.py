from truckms.service_v2.userclient.p2p_client import register_p2p_func
import traceback
import os.path as osp
import multiprocessing


def some_func(identifier, arg1, arg2, arg3):
    pass


def some_progress_func() -> dict:
    pass

def some_other_progress_func() -> dict:
    return {"a": "b"}

class p2papp:

    def __init__(self):
        self.local_port = 1000
        self.worker_pool = multiprocessing.Pool(1)
        self.list_futures = []

def test_register_p2p_func(tmpdir):
    db_url, db, col = osp.join(tmpdir, "file.txt"), "a", "b"
    app = p2papp()
    dec_func = register_p2p_func(app, db_url, db, col)(some_func)
    dec_func(identifier="a", arg1="c", arg2="d", arg3=3)

    # app.worker_pool.close()
    # app.worker_pool.join()
    app.list_futures[0].get()

    try:
        dec_func("a", arg1="c", arg2="d", arg3=3)
        assert False
    except:
        traceback.print_exc()
        assert True

    try:
        dec_func(identifier="a", arg1=["c"], arg2="d", arg3=3)
        assert False
    except:
        traceback.print_exc()
        assert True

    try:
        dec_func(identifier="a", arg1=tuple("c"), arg2="d", arg3=3)
        assert False
    except:
        traceback.print_exc()
        assert True

    try:
        dec_func(identifier="a", arg1="value_for_key_is_file", arg2="d", arg3=3)
        assert False
    except:
        traceback.print_exc()
        assert True

    try:
        dec_func(identifier="a", arg1="value_for_key_is_file", arg2="d", arg3=3)
        assert False
    except:
        traceback.print_exc()
        assert True

    file_path = osp.join(tmpdir, "file.txt")
    with open(file_path, "w") as f:
        f.write("data")
    try:
        dec_func(identifier="a", arg1="1", arg2=open(file_path, "r"), arg3=open(file_path, "r"))
        assert False
    except:
        traceback.print_exc()
        assert True

    try:
        dec_func(identifier="a", arg1="1", arg2=open(file_path, "r"), arg3=1)
        assert False
    except:
        traceback.print_exc()
        assert True

    try:
        data = open(file_path, "r")
        data.close()
        dec_func(identifier="a", arg1="1", arg2=data, arg3=1)
        assert True
    except:
        traceback.print_exc()
        assert False

    try:
        data = open(file_path, "r")
        data.close()
        dec_func(identifier="a", arg1=some_other_progress_func, arg2=some_progress_func, arg3=lambda x: {"a": "b"})
        assert True
    except:
        traceback.print_exc()
        assert False