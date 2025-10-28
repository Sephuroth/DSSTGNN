# -*- coding:utf-8 -*-
"""
Filename: layer.py
Author: lisheng
Time: 2025-04-05
"""
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolutionNetwork(nn.Module):
    def __init__(self, channels=128, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.diffusion_step = diffusion_step
        self.conv = nn.Conv2d(diffusion_step * channels, channels, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        out = []
        for i in range(0, self.diffusion_step):
            if adj.dim() == 3:
                x = torch.einsum("bcnt,bnm->bcmt", x, adj).contiguous()
                out.append(x)
            elif adj.dim() == 2:
                x = torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()
                out.append(x)
        x_cat = torch.cat(out, dim=1)
        x_conv = self.conv(x_cat)
        output = self.dropout(x_conv)
        return output


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(2, 3)
        return self.leakyrelu(e)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionNetwork(nn.Module):
    def __init__(self, n_in, n_out, dropout, alpha, nheads, order=1):
        super(GraphAttentionNetwork, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.order = order

        self.attentions = [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        for k in range(2, self.order + 1):
            self.attentions_2 = nn.ModuleList(
                [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
                 range(nheads)])

        self.out_att = GraphAttentionLayer(n_out * nheads * order, n_out, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        for k in range(2, self.order + 1):
            x2 = torch.cat([att(x, adj) for att in self.attentions_2], dim=-1)
            x = torch.cat([x, x2], dim=-1)
        x = F.elu(self.out_att(x, adj))
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        out = (input - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out


class GatedLinearUnit(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GatedLinearUnit, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()
        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 3]
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 4]
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        return time_day, time_week


class SpatialEmbedding(nn.Module):
    def __init__(self, node_geo, features):
        super(SpatialEmbedding, self).__init__()

        num_nodes = node_geo.shape[0]
        node_geo_tensor = torch.tensor(node_geo, dtype=torch.float32)
        self.linear = nn.Linear(3, features)
        self.geo_embedding = nn.Parameter(node_geo_tensor)
        self.num_nodes = num_nodes
        self.features = features

    def forward(self, batch_size):
        embedding = self.linear(self.geo_embedding)
        embedding = embedding.T.unsqueeze(0).unsqueeze(-1)
        embedding = embedding.expand(batch_size, -1, self.num_nodes, -1)

        return embedding



class TemporalAttentionModule(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TemporalAttentionModule, self).__init__()

        self.conv1 = nn.Conv2d(c_in, 1, kernel_size=(1, 1),
                               stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(num_nodes, 1, kernel_size=(1, 1),
                               stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)
        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)

        self.bn = nn.BatchNorm1d(tem_size)

    def forward(self, seq):
        seq = seq.transpose(3, 2)

        seq = seq.permute(0, 1, 3, 2).contiguous()
        c1 = seq.permute(0, 1, 3, 2)
        f1 = self.conv1(c1).squeeze()

        c2 = seq.permute(0, 2, 1, 3)
        f2 = self.conv2(c2).squeeze(axis=1)

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()

        logits = self.bn(logits).permute(0, 2, 1).contiguous()

        coefs = torch.softmax(logits, -1)
        T_coef = coefs.transpose(-1, -2)

        out = torch.einsum('bcnl,blq->bcnq', seq, T_coef)

        return out


class DualStreamTemporalExtractor(nn.Module):
    def __init__(self, features=128, layers=4, length=12, num_nodes=170, dropout=0.1):
        super(DualStreamTemporalExtractor, self).__init__()

        self.low_freq_layers = nn.ModuleList([TemporalAttentionModule(features, num_nodes, length) for _ in range(layers)])

        kernel_size = int(length / layers + 1)
        self.high_freq_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features, features, (1, kernel_size)),
                nn.ReLU(),
                nn.Dropout(dropout)) for _ in range(layers)
        ])

        self.alpha = nn.Parameter(torch.tensor(-5.0))

    def forward(self, X_low, X_high):
        res_xl = X_low
        res_xh = X_high

        for layer in self.low_freq_layers:
            X_low = layer(X_low)

        X_low = (res_xl[..., -1] + X_low[..., -1]).unsqueeze(-1)

        X_high = nn.functional.pad(X_high, (1, 0, 0, 0))

        for layer in self.high_freq_layers:
            X_high = layer(X_high)

        X_high = (res_xh[..., -1] + X_high[..., -1]).unsqueeze(-1)

        alpha_sigmoid = torch.sigmoid(self.alpha)
        output = alpha_sigmoid * X_low + (1 - alpha_sigmoid) * X_high

        return output



class DualStreamGraphLearner_layer(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        "Take in model size and number of heads."
        super(DualStreamGraphLearner_layer, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head  # We assume d_v always equals d_k
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model

        self.gcn = GraphConvolutionNetwork(channels=256, diffusion_step=1, dropout=dropout)
        self.gat = GraphAttentionNetwork(256, 256, dropout, alpha=0.2, nheads=1)

        self.LayerNorm = LayerNorm([d_model, num_nodes, seq_length], elementwise_affine=False)
        self.dropout1 = nn.Dropout(p=dropout)
        self.glu = GatedLinearUnit(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

        self.alpha = nn.Parameter(torch.tensor(-5.0))
        self.weight = nn.Parameter(torch.ones(256, self.num_nodes, 1))
        self.bias = nn.Parameter(torch.zeros(256, self.num_nodes, 1))

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 6).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(6, num_nodes).to(device), requires_grad=True).to(device)

    def forward(self, input, D_Graph):

        A_graph = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1).unsqueeze(0)
        x_gcn = self.gcn(input, A_graph)

        x_gat = self.gat(input.transpose(1, 3), D_Graph).transpose(1, 3)

        alpha_sigmoid = torch.sigmoid(self.alpha)
        x = alpha_sigmoid * x_gat + (1 - alpha_sigmoid) * x_gcn

        x = x + input
        x = self.LayerNorm(x)
        x = self.dropout1(x)
        x = self.glu(x) + x
        x = x * self.weight + self.bias + x
        x = self.LayerNorm(x)
        x = self.dropout2(x)

        return x, A_graph


class DualStreamGraphLearner(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length, dropout, num_layers):
        super(DualStreamGraphLearner, self).__init__()

        self.layers = nn.ModuleList([
            DualStreamGraphLearner_layer(device,
                      d_model=d_model,
                      head=head,
                      num_nodes=num_nodes,
                      seq_length=seq_length,
                      dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, D_Graph):
        A_graph = None
        for layer in self.layers:
            x, A_graph = layer(x, D_Graph)
        return x, A_graph