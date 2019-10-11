from truckms.inference.utils import framedatapoint_generator, framedatapoint_generator_by_frame_ids,\
    framedatapoint_generator_by_frame_ids2
import os.path as osp
from truckms.api import FrameDatapoint


def test_image_generator():
    g = framedatapoint_generator(video_path=osp.join(osp.dirname(__file__), '..', 'service' , 'data', 'cut.mkv'))
    fdp = next(g)
    assert isinstance(fdp, FrameDatapoint)


def test_image_generator_by_frame_ids():
    g = framedatapoint_generator_by_frame_ids(video_path=osp.join(osp.dirname(__file__),
                                                                  '..', 'service' , 'data', 'cut.mkv'),
                                     frame_ids=[3, 6, 10])
    fdp = next(g)
    assert fdp.frame_id == 3
    fdp = next(g)
    assert fdp.frame_id == 6
    fdp = next(g)
    assert fdp.frame_id == 10
    try:
        next(g)
        assert False
    except:
        assert True


def test_image_generator_by_frame_ids2():
    g1 = framedatapoint_generator_by_frame_ids(video_path=osp.join(osp.dirname(__file__),
                                                                   '..', 'service' , 'data', 'cut.mkv'),
                                     frame_ids=[3, 6, 10])
    g2 = framedatapoint_generator_by_frame_ids2(video_path=osp.join(osp.dirname(__file__),
                                                           '..', 'service' , 'data', 'cut.mkv'),
                                     frame_ids=[3, 6, 10])
    for fdp1, fdp2 in zip(g1, g2):
        assert fdp1.frame_id == fdp2.frame_id
        assert (fdp1.image == fdp2.image).all()
