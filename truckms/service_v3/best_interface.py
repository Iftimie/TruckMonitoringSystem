import os.path as osp
import cv2


def analysis_func(file_handler, arg_0, arg_1):
    return_data = dict()

    results_path = osp.splitext(file_handler.name)[0] + ".bin"
    with open(results_path, 'wb') as fwrite:
        fwrite.write(file_handler.read())
    return_data["results"] = open(results_path, 'rb')
    return_data["key0"] = "value0"

    return return_data


# I should also need a decorator that checks that BufferedReader arguments are closed when passed
# @p2p_register(db_url, db, coll, workload_evaluator=some_func)
def analysis_func2(video_file_handler, arg_0, arg_1, func=lambda :None):
    return_data = dict()
    name = video_file_handler.name
    video_file_handler.close()

    # progress_hook: current_index, end_index -> dictionary
    # analysis_func2.progress_hook(cur_idx, end_idx)

    cap = cv2.VideoCapture(name)
    ret, frame = cap.read()
    while ret:
        # do smth
        ret, frame = cap.read()

    with open(name, 'rb') as fread:
        results_path = osp.splitext(name)[0] + ".bin"
        with open(results_path, 'wb') as fwrite:
            fwrite.write(fread.read())
        return_data["results"] = open(results_path, 'rb')
        return_data["key0"] = "value0"

    # I should decide whether the BufferedReader should be closed or not when it is returned.
    # for sanity purposes. both when it is passed, and when it is returned it should be closed

    return return_data

collection = []

file_handler = open(osp.join(osp.dirname(__file__), "..", "..", "tests", "truckms", "service", "data", "cut.mkv"), "rb")

input_data = {"file_handler": file_handler, "arg_0": 0, "arg_1": 1}
collection.append(input_data)
returned_data = analysis_func(**input_data)
collection[0].update(returned_data)

input_data2 = {"video_file_handler": file_handler, "arg_0": 0, "arg_1": 1}
collection.append(input_data2)
returned_data2 = analysis_func2(**input_data2)
collection[1].update(returned_data2)


# from userclient.py
for item in collection:
    analysis_func2.promise_get(item)


#broker
functions_list = [analysis_func2]

#clientworker
# do_work(analysis_func2)