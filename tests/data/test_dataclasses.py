from fast_gcn.data._dataclasses import *


def test_remove_bodies(bodies):
    num_bodies = bodies.num_bodies
    body_ids = bodies.body_ids

    bodies.remove_body(body_ids[0])
    assert bodies.num_bodies == num_bodies - 1
