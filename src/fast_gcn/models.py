import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric.nn as geom_nn
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

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: N, T, M, V, C
        N, T, M, V, C = x.size()
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
        one_hot_joint = one_hot_joint.view(1, 1, 1, num_joints, num_joints)
        self.register_buffer("j", one_hot_joint)

        self.embed_joint = Embed([num_joints, 64, 64])

        self.compute_adjacency_matrix = AdjacencyMatrix(num_joints, 128, 256)

        self.gcn1 = self.build_gcn(128, 128)
        self.gcn2 = self.build_gcn(128, 256)
        self.gcn3 = self.build_gcn(256, 256)

    def forward(self, z):
        # z: N, T, M, V, C
        N, T, M, V, C = z.size()
        z = torch.cat([z, self.embed_joint(self.j).repeat(N, T, M, 1, 1)], dim=-1)

        # G: N, T, M, V, V
        G = self.compute_adjacency_matrix(z)

        x = z.view(N * T * M, V, -1)
        adj = G.view(N * T * M, V, V)

        # x: N, V, C, adj: N, V, V
        x = self.gcn1(x, adj, mask=None, add_loop=False)
        x = self.gcn2(x, adj, mask=None, add_loop=False)
        x = self.gcn3(x, adj, mask=None, add_loop=False)

        z = x.view(N, T, M, V, -1)
        return z

    def build_gcn(self, in_channels, out_channels):
        return Sequential(
            "x, adj, mask, add_loop",
            [
                (
                    geom_nn.DenseGCNConv(in_channels, out_channels),
                    "x, adj, mask, add_loop -> x",
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

        self.embed_frame = Embed([length, 64, 256])

        self.tcn1 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.tcn2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, z):
        # z: N, T, M, V, C
        N, T, M, V, C = z.size()
        z = z + self.embed_frame(self.f)
        z = z.max(dim=-2).values
        z = z.squeeze(-2)  # spatial max pooling

        # z: N, T, M, C
        x = z.permute(0, 2, 3, 1).reshape(N * M, C, T)
        # x: N, C, T
        x = self.tcn1(x)
        x = self.tcn2(x)

        _, C, T = x.size()
        # x: N, C, T
        x = x.view(N, M, C, T)
        # x: N, M, T, C
        z = x.max(dim=-1).values  # temporal max pooling
        z = z.squeeze(-2)

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
        # z: N, T, M, V, C
        theta = self.theta(z)
        phi = self.phi(z)

        S = torch.matmul(theta, phi.permute(0, 1, 2, 4, 3).contiguous())
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
