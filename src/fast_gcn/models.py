import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.nn import Sequential

from typing import List


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

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: N, T, M, V, C
        N, T, M, V, C = x.size()
        z = self.dynamics_representation(x)
        # z: N, T, M, V, C
        z = self.frame_level_module(z)
        # z: N, M, V, C
        z = self.joint_level_module(z)
        # z: N, M, C
        z = z.mean(dim=1)
        # z: N, C
        logits = self.fc(z)
        return logits


class DynamicsRepresentation(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.embed_pos = Embed([num_features, 64, 64])
        self.embed_vel = Embed([num_features, 64, 64])

    def forward(self, joint):
        N, T, M, V, C = joint.size()
        padded_position = torch.cat(
            [joint, torch.zeros(N, 1, M, V, C).to(joint.device)], dim=1
        )
        velocity = padded_position[:, 1:] - padded_position[:, :-1]

        vel_embedding = self.embed_vel(velocity)
        pos_embedding = self.embed_pos(joint)

        fused_embedding = pos_embedding + vel_embedding

        return fused_embedding


class JointLevelModule(nn.Module):
    def __init__(self, num_joints, num_features):
        super().__init__()
        joint_indices = torch.arange(0, num_joints)
        one_hot_joint = F.one_hot(joint_indices, num_classes=num_joints).float()
        one_hot_joint = one_hot_joint.view(1, 1, num_joints, num_joints)
        self.register_buffer("j", one_hot_joint)

        self.embed_joint = Embed([num_joints, 64, 64])

        self.compute_adjacency_matrix = AdjacencyMatrix(num_joints, 128, 256)

        self.gcn1 = self.build_gcn(128, 128)
        self.gcn2 = self.build_gcn(128, 256)
        self.gcn3 = self.build_gcn(256, 256)

    def forward(self, z):
        # z: N, M, V, C
        N, M, V, C = z.size()
        z = torch.cat([z, self.embed_joint(self.j).repeat(N, M, 1, 1)], dim=-1)

        G = self.compute_adjacency_matrix(z)

        # G: N, M, V, V
        adj = G.view(N * M, V, V)

        x = z.view(N * M, V, -1)
        # x: N, V, C, adj: N, V, V
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        x = self.gcn3(x, adj)

        z = x.view(N, M, V, -1)

        # z: N, M, V, C
        z = z.max(dim=-2).values
        # z: N, M, 1, C
        z = z.squeeze(-2)  # spatial max pooling
        # x: N, M, C

        return z

    def build_gcn(self, in_channels, out_channels):
        return Sequential(
            "x, adj",
            [
                (
                    GCN(in_channels, out_channels),
                    "x, adj -> x",
                ),
                GraphBatchNorm(out_channels),
                nn.ReLU(inplace=True),
            ],
        )


class FrameLevelModule(nn.Module):
    def __init__(self, length):
        super().__init__()
        frame_indices = torch.arange(0, length)
        one_hot_frame = F.one_hot(frame_indices, num_classes=length).float()
        one_hot_frame = one_hot_frame.view(1, length, 1, 1, length)
        self.register_buffer("f", one_hot_frame)

        self.embed_frame = Embed([length, 64, 64])

        self.tcn1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.drop_out = nn.Dropout(0.2)

        self.tcn2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, z):
        # z: N, T, M, V, C
        N, T, M, V, C = z.size()
        z = z + self.embed_frame(self.f)

        # z: N, T, M, V, C
        x = z.permute(0, 2, 3, 4, 1).reshape(N * M * V, C, T)
        # x: N, C, L
        x = self.tcn1(x)
        x = self.drop_out(x)
        x = self.tcn2(x)

        _, C, T = x.size()
        # x: N, C, T
        x = x.view(N, M, V, C, T).permute(0, 4, 1, 2, 3).contiguous()
        # x: N, T, M, V, C
        z = x.max(dim=1).values  # temporal max pooling
        z = z.squeeze(1)
        # z: N, M, V, C

        return z


class Embed(nn.Module):
    def __init__(self, feature_dims: List[int] = [64, 64, 64]):
        super().__init__()

        modules = []
        for in_channels, out_channels in zip(feature_dims[:-1], feature_dims[1:]):
            module = self.build_fc(in_channels, out_channels)
            modules.append(module)

        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        return self.fc(x)

    def build_fc(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.ReLU(inplace=True)
        )


class AdjacencyMatrix(nn.Module):
    def __init__(self, num_joints, in_features, out_features):
        super().__init__()

        self.theta = nn.Linear(in_features, out_features)
        self.phi = nn.Linear(in_features, out_features)

        self.softmax = nn.Softmax(dim=-2)

    def forward(self, z):
        # z: N, M, V, C
        theta = self.theta(z)
        phi = self.phi(z)

        S = torch.matmul(theta, phi.permute(0, 1, 3, 2).contiguous())
        G = self.softmax(S)
        return G


class GraphBatchNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, x):
        N, V, C = x.size()
        x = x.permute(0, 2, 1).contiguous()
        # x: N, C, V
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1).contiguous()
        # x: N, V, C
        return x


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.lin1 = Linear(
            in_channels, out_channels, bias=False, weight_initializer="glorot"
        )

        self.lin2 = Linear(
            in_channels, out_channels, bias=False, weight_initializer="glorot"
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        zeros(self.bias)

    def forward(self, x, adj, mask=None):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        out = self.lin1(x)

        out = torch.matmul(adj, out)

        out = out + self.lin2(x)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, " f"{self.out_channels})"
