import torch

from fast_gcn.models import STSGN


def test_stsgn():
    num_classes = 4
    length = 8
    num_joints = 5
    num_features = 2
    x = torch.randn(num_classes, length, 3, num_joints, num_features)

    model = STSGN(num_classes, length, num_joints, num_features)

    logits = model(x)

    assert logits.size() == (4, num_classes)
