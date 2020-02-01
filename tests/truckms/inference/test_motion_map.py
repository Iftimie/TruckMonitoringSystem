import os.path as osp
from truckms.inference.motion_map import movement_frames_indexes
from truckms.inference.utils import get_video_file_size


def test_motion_map_generator():

    video_test = osp.join(osp.dirname(__file__), "..", "service", "data", "cut.mkv")

    list_movement = [movement for movement in movement_frames_indexes(video_test)]
    assert get_video_file_size(video_test) == len(list_movement)


def test_movement_frames():
    video_test = osp.join(osp.dirname(__file__), "..", "service", "data", "cut.mkv")

    result = movement_frames_indexes(video_test)
    assert len(result) <= get_video_file_size(video_test)