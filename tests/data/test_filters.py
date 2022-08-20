from fast_gcn.data import filters


def test_filter_by_motion(bodies, max_bodies):
    bodies = filters.filter_by_motion(bodies, 1)

    assert bodies.num_bodies <= 1


def test_pad_frames(bodies, max_bodies, num_joints, num_features):
    max_bodies = 4
    frames = filters.pad_frames(bodies.to_list(), max_bodies, num_joints, num_features)

    assert frames.size()[1:] == (max_bodies, num_joints, num_features)
