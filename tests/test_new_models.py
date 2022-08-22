import pytest

import torch

from fast_gcn.models import SemanticSTGCN


@pytest.fixture
def x():
    return torch.randn(1, 30, 2, 25, 3)


@pytest.fixture
def length():
    return 30


@pytest.fixture
def num_joints():
    return 25


@pytest.fixture
def num_features():
    return 3


def test_semantic_stgcn(x, length, num_joints, num_features):
    num_classes = 5
    model = SemanticSTGCN(num_classes, length, num_joints, num_features, emb_dim=64)

    logits = model(x)

    assert logits.size() == (x.size(0), num_classes)
