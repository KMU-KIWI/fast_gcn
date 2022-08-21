import pytest

import torch

from fast_gcn.data import samplers


@pytest.mark.parametrize("length", [1, 10, 20])
def test_sample_frames(length):
    x = torch.randn(10, 2, 25, 3)
    x = samplers.sample_frames(x, length)

    assert x.size() == (length, 2, 25, 3), print(x.size())
