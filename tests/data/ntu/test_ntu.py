import pytest

from torch_geometric.loader import DataLoader

from fast_gcn.data import NTU60, NTU120


"""
@pytest.mark.parametrize("benchmark", ["xsub", "xview"])
@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("joint_type", ["3d", "color_2d", "depth_2d"])
@pytest.mark.parametrize("max_bodies", [4])
def test_ntu60(ntu_path, benchmark, split, joint_type, max_bodies):
    dataset = NTU60(
        root=ntu_path,
        benchmark=benchmark,
        split=split,
        joint_type=joint_type,
        max_bodies=max_bodies,
    )
    data = dataset[0]

    dataloader = DataLoader(dataset)
    for batch in dataloader:
        break
"""


@pytest.mark.parametrize("benchmark", ["xsub", "xview"])
@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("joint_type", ["3d", "color_2d", "depth_2d"])
@pytest.mark.parametrize("max_bodies", [4])
def test_ntu120(ntu_path, benchmark, split, joint_type, max_bodies):
    dataset = NTU120(
        root=ntu_path,
        benchmark=benchmark,
        split=split,
        joint_type=joint_type,
        max_bodies=max_bodies,
    )
    data = dataset[0]
    print(data)

    dataloader = DataLoader(dataset)
    for batch in dataloader:
        print(batch)
        break
