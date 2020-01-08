from truckms.service_v2.api import validate_arguments
import os.path as osp


# def test_validate_arguments(tmpdir):
#     file_path = osp.join(tmpdir, "file.txt")
#     with open(file_path, "w") as f:
#         f.write("data")
#
#     fd = open(file_path, 'rb')
#     validate_arguments(tuple(fd, "a"))