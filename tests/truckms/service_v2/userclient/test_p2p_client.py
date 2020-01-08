from truckms.service_v2.userclient.p2p_client import register_p2p_func, create_p2p_client_app
import traceback
import os.path as osp
import multiprocessing
import tinymongo


def some_func(identifier, arg1, arg2, arg3) -> dict:
    ret_value = {"value": "{},{},{},{}".format(identifier, arg1, arg2, arg3)}
    return ret_value


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
    db_url, db, col = osp.join(tmpdir, "dburl"), "a", "b"
    app = create_p2p_client_app()
    dec_func = app.register_p2p_func(db_url, db, col)(some_func)
    dec_func(identifier="a", arg1="c", arg2="d", arg3=3)

    # app.worker_pool.close()
    # app.worker_pool.join()
    print(app.list_futures[0].get())

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


def func_return_dict(arg1, arg2) -> dict:
    return {arg1: arg2}


def func_return_val():
    return 10


def complex_func(identifier, int_arg, str_arg, file_arg, func_arg, func_arg_ret_dict) -> dict:
    ret_dict = func_arg_ret_dict("func_arg_ret_dict_key", 10)
    assert ret_dict is not None
    val = "{},{},{},{}".format(identifier, int_arg, str_arg, func_arg(), file_arg.name)
    ret_dict = {"val": val}
    return ret_dict


def test_register_p2p_func2(tmpdir):
    file_path = osp.join(tmpdir, "file.txt")
    with open(file_path, "w") as f:
        f.write("data")

    db_url, db, col = osp.join(tmpdir, "dburl"), "db", "col"
    app = create_p2p_client_app()
    decorated_func = app.register_p2p_func(db_url, db, col)(complex_func)

    try:
        data = open(file_path, "r")
        data.close()
        decorated_func(identifier="x", int_arg=10, str_arg="str", file_arg=data, func_arg=func_return_val, func_arg_ret_dict=func_return_dict)
        assert True
        app.worker_pool.close()
        app.worker_pool.join()
        item = tinymongo.TinyMongoClient(db_url)[db][col].find({"identifier": "x"})[0]
        assert item['identifier'] == 'x'
        assert item['int_arg'] == 10
        assert item['str_arg'] == 'str'
        assert item['nodes'] == []
        assert item['current_address'] is None
        assert item['file_arg'] == file_path
        assert item['func_arg_ret_dict_key'] == 10
        assert item['val'] == 'x,10,str,10'
    except:
        traceback.print_exc()
        assert False