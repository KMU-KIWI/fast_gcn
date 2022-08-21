import torch
from torch import nn
import torch.nn.functional as F

from .sgn import Embed, GCN


class STSGNBlock(nn.Module):
    def __init__(
        self,
        num_joints,
        length,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
    ):
        super().__init__()
        joint_indices = torch.arange(0, num_joints)
        one_hot_joint = F.one_hot(joint_indices, num_classes=num_joints).float()
        one_hot_joint = one_hot_joint.view(1, 1, 1, num_joints, num_joints)
        self.register_buffer("j", one_hot_joint)

        self.embed_joint = Embed([num_joints, 64, in_channels])

        self.gcn = GCN(2 * in_channels, out_channels)

        frame_indices = torch.arange(0, length)
        one_hot_frame = F.one_hot(frame_indices, num_classes=length).float()
        one_hot_frame = one_hot_frame.view(1, length, 1, 1, length)
        self.register_buffer("f", one_hot_frame)

        self.embed_frame = Embed([length, 64, out_channels])

        self.tcn = nn.Sequential(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.adj_tcn = nn.Conv1d(
            num_joints,
            num_joints,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x, G):
        # x: N, T, M, V, C adj: N, T, M, V, V
        N, T, M, V, C = x.size()

        x = torch.cat([x, self.embed_joint(self.j).repeat(N, T, M, 1, 1)], dim=-1)
        x = x.view(N * T * M, V, -1)
        adj = G.view(N * T * M, V, V)

        # adj: N, V, V
        # x: N, V, C
        x = self.gcn(x, adj)
        C = x.size(-1)
        x = x.view(N, T, M, V, C)

        # x: N, T, M, V, C
        x = x + self.embed_frame(self.f)
        x = x.permute(0, 2, 3, 4, 1).reshape(N * M * V, C, T)
        G = G.permute(0, 2, 3, 4, 1).reshape(N * M * V, V, T)
        # x: N, C, L
        x = self.tcn(x)
        G = self.adj_tcn(G)

        _, C, T = x.size()
        x = x.view(N, M, V, C, T).permute(0, 4, 1, 2, 3).contiguous()
        G = G.view(N, M, V, V, T).permute(0, 4, 1, 2, 3).contiguous()
        # x: N, T, M, V, C
        return x, G
