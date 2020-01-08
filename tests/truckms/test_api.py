from truckms.api import FrameDatapoint, PredictionDatapoint
import numpy as np


def test_api():
    def create_frame_datapoint() -> FrameDatapoint:
        return FrameDatapoint(np.zeros((100, 100), dtype=np.uint8), 0)

    fdp = create_frame_datapoint()
    assert isinstance(fdp, FrameDatapoint)
    assert fdp.frame_id == 0

    def func_with_dpf(arg: FrameDatapoint) -> FrameDatapoint:
        arg.frame_id = 100
        return arg

    new_dfp = func_with_dpf(arg=fdp)
    assert new_dfp.frame_id == 100


