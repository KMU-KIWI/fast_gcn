from torch import nn

from torch_geometric.nn import Sequential

from fast_gcn.layers import (
    STSGNBlock,
    AdjacencyMatrix,
    DynamicsRepresentation,
)


class STSGN(nn.Module):
    def __init__(self, num_classes, length, num_joints, num_features):
        super().__init__()

        self.num_classes = num_classes
        self.length = length
        self.num_joints = num_joints
        self.num_features = num_features

        self.dynamics_representation = DynamicsRepresentation(num_features)
        self.compute_adjacency_matrix = AdjacencyMatrix(num_joints, 64, 256)

        self.features = Sequential(
            "x, adj",
            [
                (
                    STSGNBlock(
                        num_joints,
                        length,
                        64,
                        128,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    "x, adj -> x, adj",
                ),
                (
                    STSGNBlock(
                        num_joints,
                        length // 2,
                        128,
                        256,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    "x, adj -> x, adj",
                ),
                (
                    STSGNBlock(
                        num_joints,
                        length // 2,
                        256,
                        256,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    "x, adj -> x, adj",
                ),
            ],
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: N, T, M, V, C
        N, T, M, V, C = x.size()
        x = self.dynamics_representation(x)
        # x: N, T, M, V, C
        G = self.compute_adjacency_matrix(x)
        # G: N, T, M, V, V
        x = self.features(x, G)
        # x: N, T, M, V, C
        x = x.mean(dim=1).mean(dim=1).mean(dim=1)
        # z: N, C
        logits = self.fc(x)
        return logits
