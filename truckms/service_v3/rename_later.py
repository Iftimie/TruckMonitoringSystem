import os.path as osp

# p2p_broker.register()

# p2p_client_worker.register()

# p2p_client.register()

# I should also need a decorator that checks that BufferedReader arguments are closed when passed
# also check that arguments are not instances of functions
# @p2p_register(db_url, db, coll, workload_evaluator=some_func)
def func(file_handler, arg_0, arg_1, arg_2, other_func=lambda: None):
    return_data = dict()
    name = file_handler.name
    file_handler.close()

    # progress_hook: current_index, end_index -> dictionary
    # analysis_func2.progress_hook(cur_idx, end_idx)

    with open(name, 'rb') as fread:
        results_path = osp.splitext(name)[0] + ".bin"
        with open(results_path, 'wb') as fwrite:
            fwrite.write(fread.read())
        return_data["results"] = open(results_path, 'rb')
        return_data["key0"] = "value0"

    # I should decide whether the BufferedReader should be closed or not when it is returned.
    # for sanity purposes. both when it is passed, and when it is returned it should be closed

    return return_data

def test_func(tmpdir):
    file_path = osp.join(tmpdir, "file.txt")
    with open(file_path, "w") as f:
        f.write("data")
    func(open(file_path, 'rb'), 0, "1", {"a": "b", 1: 2}, other_func=lambda )
