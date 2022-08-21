import torch.nn as nn

class norm_data(nn.Module):
    def __init__(self, dim= 64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim* 25)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x

class embed(nn.Module):
    def __init__(self, dim = 3, dim1 = 128, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x

class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

class local(nn.Module):
    def __init__(self, dim1 = 3, dim2 = 3, bias = False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias = False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)


    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x

class compute_g_spa(nn.Module):
    def __init__(self, dim1 = 64 *3, dim2 = 64*3, bias = False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):

        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g


class SpatialGAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_subset=8, num_node=25, num_frame=32,
                 kernel_size=1, stride=1, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=False,
                 use_temporal_att=False, use_spatial_att=True, attentiondrop=0, use_pes=True, use_pet=False,
                 residual=True):
        super(SpatialGAT, self).__init__()
        inter_channels = out_channels // num_subset
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.glo_reg_t = glo_reg_t
        self.att_t = att_t
        self.use_pes = use_pes
        self.use_pet = use_pet
        self.window_size = 3
        self.residual = residual

        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att
        self.out_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (7, 1), padding=(3, 0), bias=True, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )
        if use_spatial_att:
            atts = torch.zeros((1, num_subset, num_node * self.window_size, num_node))
            self.register_buffer('atts', atts)
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_s:
                self.in_nets = nn.Conv2d(in_channels, num_subset * inter_channels, 1, bias=True)
                self.in_nets_upfold = nn.Conv2d(in_channels, num_subset * inter_channels, 1, bias=True)
                self.upfold = UnfoldTemporalWindows(self.window_size, 1, 1)
                self.diff_net = nn.Conv2d(in_channels, in_channels, 1, bias=True)

                from ntu_rgb_d import AdjMatrixGraph
                from torch.autograd import Variable
                import numpy as np
                self.graph = Variable(torch.from_numpy(AdjMatrixGraph().A_sep.astype(np.float32)), requires_grad=False)
                self.graph_a = Variable(torch.from_numpy(AdjMatrixGraph().A.astype(np.float32)), requires_grad=False)
            if glo_reg_s:
                self.attention0s = nn.Parameter(
                    torch.ones(1, num_subset, num_node * self.window_size, num_node) / num_node,
                    requires_grad=True)

            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        if in_channels != out_channels or stride != 1:
            if use_spatial_att:
                self.downs1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )

            if use_temporal_att:
                self.downt1 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downt2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            if use_spatial_att:
                self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            if use_temporal_att:
                self.downt1 = lambda x: x
            self.downt2 = lambda x: x

        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)
        self.resi_relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = x.permute(0, 1, 3, 2)
        identity = x
        N, C, T, V = x.size()
        # print(self.betas)
        if self.use_spatial_att:
            attention = self.atts

            y = x
            if self.att_s:
                upfold = self.upfold(y)
                k = self.in_nets(y).view(N, self.num_subset, self.inter_channels, T, V)  # nctv -> n num_subset c'tv
                q = self.in_nets_upfold(upfold).view(N, self.num_subset, self.inter_channels, T, self.window_size * V)

                import torch.nn.functional as F
                attention = attention + torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)

                zero_vec = -9e15 * torch.ones_like(attention[:, 0])

                self.graph = self.graph.cuda()
                attention = torch.cat([torch.where(self.graph[i].repeat(self.window_size, 1) > 0, attention[:, i, :, :],
                                                   zero_vec).unsqueeze(1) for i in range(self.num_subset)], 1)
                # attention = F.softmax(attention, dim=-2)* self.alphas
                attention = F.softmax(attention, dim=-1)  # * self.betas.repeat(1,1,25,1)

                self.graph_a = self.graph_a.cuda()
                # diff = (self.diff_net(torch.einsum('nctu,uv->nctv',y, self.graph_a)).repeat(1,1,1,self.window_size) - upfold)\
                #     .view(N, self.num_subset, self.in_channels//self.num_subset, T ,self.window_size, V)  #nsctwv
                diff = (self.diff_net(torch.einsum('nctu,uv->nctv', y, self.graph_a)).repeat(1, 1, 1,
                                                                                             self.window_size) - upfold) \
                    .view(N, self.num_subset, -1, T, self.window_size, V)  # nsctwv
                diff = torch.sigmoid(diff.mean(-1).mean(2).mean(2))  # nsw
                attention = attention * diff.unsqueeze(-1).repeat(1, 1, 25, 1)
            if self.glo_reg_s:
                attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            y = torch.einsum('nctu,nsuv->nsctv', upfold, attention).contiguous().view(N,
                                                                                      self.num_subset * self.in_channels,
                                                                                      T, V)

            y = self.out_nets(y)  # nctv

            # y = self.out_conv(y.view(N, self.out_channels, T, -1, V)).squeeze(3)

            y = self.relu(self.downs1(x) + y)

            y = self.ff_nets(y)

            y = self.relu(self.downs2(x) + y)
        else:
            y = self.out_nets(x)
            y = self.relu(self.downs2(x) + y)


        if self.residual:
            z = y
            z += identity
            z = self.resi_relu(z)
            z = z.permute(0, 1, 3, 2)
        else:
            z = y
            z = z.permute(0, 1, 3, 2)
        return z


class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size, window_stride, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size - 1) * (window_dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    # @autocast()
    def forward(self, x):
        # Input shape: (N,C,T,V), out: (N,C,T,V*window_size)
        N, C, T, V = x.shape
        x = self.unfold(x)  # (N, C*Window_Size, (T-Window_Size+1)*(V-1+1))
        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        x = x.view(N, C, self.window_size, -1, V).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(N, C, -1, self.window_size * V)
        return x