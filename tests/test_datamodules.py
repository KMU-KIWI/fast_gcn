import pytest

import fast_gcn.datamodules as dm


@pytest.mark.parametrize("benchmark", ["xsub", "xview"])
@pytest.mark.parametrize("joint_type", ["3d", "color_2d", "depth_2d"])
@pytest.mark.parametrize("max_bodies", [1, 5])
def test_ntu_datamodule(ntu_path, benchmark, joint_type, max_bodies):
    data = dm.NTU60DataModule(
        batch_size=3,
        length=100,
        data_dir=ntu_path,
        benchmark=benchmark,
        joint_type=joint_type,
        max_bodies=max_bodies,
    )

    data.setup()

    dataloader = data.train_dataloader()

    for x, edge_index, y in dataloader:
        break

    assert x.size() == (3, 100, max_bodies, 25, 3 if joint_type == "3d" else 2)
