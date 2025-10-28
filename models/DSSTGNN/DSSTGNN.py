# -*- coding:utf-8 -*-
"""
Filename: DSSTGNN.py
Author: lisheng
Time: 2025-04-06
"""

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DSSTGNN.layer import TemporalEmbedding, DualStreamTemporalExtractor,DualStreamGraphLearner, SpatialEmbedding

class DSSTGNN(nn.Module):
    def __init__(
            self, device, batch_size=64, input_dim=5, channels=64, num_nodes=156, input_len=12, output_len=12, dropout=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.head = 1
        self.layers = 2
        self.dims = 6

        if num_nodes == 156:
            time = 288
            self.layers = 2

        self.Temb = TemporalEmbedding(time, channels)
        self.start_conv_res = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))
        self.start_conv_1 = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))
        self.start_conv_2 = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))
        self.network_channel = channels * 2

        self.DSTE = DualStreamTemporalExtractor(
            features=128,
            layers=self.layers,
            length=self.input_len,
            num_nodes=self.num_nodes,
            dropout=0.1
        )

        self.DSGL = DualStreamGraphLearner(
            device,
            d_model=self.network_channel,
            head=self.head,
            num_nodes=num_nodes,
            seq_length=1,
            dropout=dropout,
            num_layers=self.layers
        )

        self.MLP = nn.Conv2d(
            in_channels=5,
            out_channels=self.dims,
            kernel_size=(1, 1)
        )


        stationsa_data = np.load('data/station_metadata.npy')
        self.spatial_embedding_layer = SpatialEmbedding(node_geo=stationsa_data, features=self.dims)
        self.E_s = self.spatial_embedding_layer(self.batch_size).to(device)

        self.fc_d = nn.Conv2d(channels, self.dims, kernel_size=(1, 1))
        self.fc_w = nn.Conv2d(channels, self.dims, kernel_size=(1, 1))

        self.fc_st = nn.Conv2d(self.network_channel, self.network_channel, kernel_size=(1, 1))

        self.regression_layer = nn.Conv2d(self.network_channel, self.output_len * 3, kernel_size=(1, 1))

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        input_data = history_data
        residual_cpu = input_data.cpu()
        residual_numpy = residual_cpu.detach().numpy()
        coef = pywt.wavedec(residual_numpy, 'db1', level=2)
        coefl = [coef[0]] + [None] * (len(coef) - 1)
        coefh = [None] + coef[1:]
        xl = pywt.waverec(coefl, 'db1')
        xh = pywt.waverec(coefh, 'db1')

        xl = torch.from_numpy(xl).to(self.device)
        xh = torch.from_numpy(xh).to(self.device)

        input_data_1 = self.start_conv_1(xl)
        input_data_2 = self.start_conv_2(xh)

        input_data = self.DSTE(input_data_1, input_data_2)
        E_tod, E_dow = self.Temb(history_data.permute(0, 3, 2, 1))
        TE = E_tod + E_dow

        E_d = torch.tanh(self.MLP(history_data)[..., -1].unsqueeze(-1) *
                         (self.fc_d(self.Temb(history_data.permute(0, 3, 2, 1))[0]) *
                          self.fc_w(self.Temb(history_data.permute(0, 3, 2, 1))[1])) *
                         self.E_s)[-1, ..., -1]

        D_graph = F.softmax(F.relu(torch.mm(E_d.transpose(0, 1), E_d)), dim=1)

        data_st = torch.cat([input_data] + [TE], dim=1)

        skip = self.fc_st(data_st)
        out, A_graph = self.DSGL(data_st, D_graph)
        data_st = out + skip

        prediction = self.regression_layer(data_st)
        prediction = prediction.view(self.batch_size, self.output_len, 3, self.num_nodes, 1)
        prediction = prediction.permute(0, 2, 3, 1, 4).squeeze(-1)

        return prediction, A_graph, D_graph
