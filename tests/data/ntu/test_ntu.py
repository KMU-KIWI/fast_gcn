import pytest

import os

from torch.utils.data import DataLoader

from fast_gcn.data import NTU60, NTU120
from fast_gcn.transforms import SampleFrames


@pytest.mark.parametrize("benchmark", ["xsub", "xview"])
@pytest.mark.parametrize("split", ["val", "train"])
@pytest.mark.parametrize("joint_type", ["3d", "color_2d", "depth_2d"])
@pytest.mark.parametrize("max_bodies", [1, 5])
def test_ntu60(ntu_path, benchmark, split, joint_type, max_bodies):
    transform = SampleFrames(length=100)
    dataset = NTU60(
        root=ntu_path,
        benchmark=benchmark,
        split=split,
        joint_type=joint_type,
        max_bodies=max_bodies,
        transform=transform,
        download=True,
    )

    x, edge_index, y = dataset[0]
    statement = x.size() == (100, max_bodies, 25, 3 if joint_type == "3d" else 2)

    if statement is False:
        os.remove(dataset.processed_file_paths[0])

    assert statement

    transform = SampleFrames(length=100)
    dataset = NTU60(
        root=ntu_path,
        benchmark=benchmark,
        split=split,
        joint_type=joint_type,
        max_bodies=max_bodies,
        transform=transform,
        download=True,
    )

    dataloader = DataLoader(dataset, batch_size=4)

    for x, edge_index, y in dataloader:
        break

    os.remove(dataset.processed_file_paths[0])

    assert x.size() == (4, 100, max_bodies, 25, 3 if joint_type == "3d" else 2)


@pytest.mark.parametrize("benchmark", ["xsub", "xview"])
@pytest.mark.parametrize("split", ["val", "train"])
@pytest.mark.parametrize("joint_type", ["3d", "color_2d", "depth_2d"])
@pytest.mark.parametrize("max_bodies", [1, 5])
def test_ntu120(ntu_path, benchmark, split, joint_type, max_bodies):
    transform = SampleFrames(length=100)
    dataset = NTU120(
        root=ntu_path,
        benchmark=benchmark,
        split=split,
        joint_type=joint_type,
        max_bodies=max_bodies,
        transform=transform,
        download=True,
    )
    x, edge_index, y = dataset[0]

    statement = x.size() == (100, max_bodies, 25, 3 if joint_type == "3d" else 2)

    if statement is False:
        os.remove(dataset.processed_file_paths[0])

    assert statement

    dataset = NTU120(
        root=ntu_path,
        benchmark=benchmark,
        split=split,
        joint_type=joint_type,
        max_bodies=max_bodies,
        transform=transform,
        download=True,
    )

    dataloader = DataLoader(dataset, batch_size=4)

    for x, edge_index, y in dataloader:
        break

    os.remove(dataset.processed_file_paths[0])

    assert x.size() == (4, 100, max_bodies, 25, 3 if joint_type == "3d" else 2)
