from torch import nn

from fast_gcn.layers import DynamicsRepresentation, JointLevelModule, FrameLevelModule


class SGN(nn.Module):
    def __init__(self, num_classes, length, num_joints, num_features):
        super().__init__()

        self.num_classes = num_classes
        self.length = length
        self.num_joints = num_joints
        self.num_features = num_features

        self.dynamics_representation = DynamicsRepresentation(num_features)

        self.joint_level_module = JointLevelModule(num_joints, num_features)
        self.frame_level_module = FrameLevelModule(length)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: N, T, M, V, C
        z = self.dynamics_representation(x)
        # z: N, T, M, V, C
        z = self.joint_level_module(z)
        # z: N, T, M, V, C
        z = self.frame_level_module(z)
        # z: N, M, C
        z = z.mean(dim=1)
        # z: N, C
        logits = self.fc(z)
        return logits
