import pytest

import torch

from fast_gcn.data.utils import Bodies


J = 25
C = 3


@pytest.fixture
def bodies():
    frames = (
        [
            dict(
                a=torch.randn(J, C),
                c=torch.randn(J, C),
            )
        ]
        * 8
        + [
            dict(
                a=torch.randn(J, C),
                b=torch.randn(J, C),
                c=torch.randn(J, C),
            )
        ]
        * 8
        + [
            dict(
                c=torch.randn(J, C),
            )
        ]
        * 8
    )

    return Bodies.from_list(frames)


@pytest.fixture
def max_bodies():
    return 2


@pytest.fixture
def num_joints():
    return J


@pytest.fixture
def num_features():
    return C
