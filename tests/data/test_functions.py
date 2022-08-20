import pytest

import torch

import fast_gcn.data.functions as F


def test_filter_by_motion(bodies, max_bodies):
    bodies = F.filter_by_motion(bodies, 1)

    assert bodies.num_bodies <= 1


def test_pad_frames(bodies, max_bodies, num_joints, num_features):
    max_bodies = 4
    frames = F.pad_frames(bodies.to_list(), max_bodies, num_joints, num_features)

    assert frames.size()[1:] == (max_bodies, num_joints, num_features)


@pytest.mark.parametrize("length", [1, 10, 20])
def test_sample_frames(length):
    x = torch.randn(10, 2, 25, 3)
    x = F.sample_frames(x, length)

    assert x.size() == (length, 2, 25, 3)
