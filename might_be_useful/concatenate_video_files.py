import os
from truckms.inference.utils import create_avi, framedatapoint_generator

input_files = [os.path.join(r"D:\tms_data\vidin_data", f) for f in os.listdir(r"D:\tms_data\vidin_data") if ".mp4" in f and "concatenated" not in f]
destination_file = r"D:\tms_data\vidin_data\concatenated.avi"

generators = [framedatapoint_generator(video_path, skip=0) for video_path in input_files]
first_frame = None
while True:
    try:
        first_frame = next(generators[0])
        break
    except:
        generators.pop(0)

with create_avi(destination_file, first_image=first_frame.image) as append_fn:
    for g in generators:
        for fdp in g:
            append_fn(fdp.image)
