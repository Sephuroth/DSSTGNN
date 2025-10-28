# -*- coding:utf-8 -*-
"""
Filename: engine.py
Author: lisheng
Time: 2025-04-06
"""

import util
from models.DSSTGNN.DSSTGNN import DSSTGNN
from optimizer import Ranger
from util import *

class trainer:
    def __init__(
        self,
        scaler,
        batch_size,
        input_dim,
        channels,
        num_nodes,
        input_len,
        output_len,
        dropout,
        lrate,
        wdecay,
        device,
    ):
        self.model = DSSTGNN(device, batch_size, input_dim, channels, num_nodes, input_len, output_len, dropout)
        self.model.to(device)
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.MAE_torch   # MAE_Freq_torch
        self.scaler = scaler
        self.clip = 5
        print(self.model)

    def train(self, input, real_val):

        self.model.train()
        self.optimizer.zero_grad()
        output, A_graph, D_graph = self.model(input)
        real = real_val
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape, A_graph.detach(), D_graph.detach()

    def eval(self, input, real_val):
        self.model.eval()
        output, A_graph, D_graph = self.model(input)
        real = real_val
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape, A_graph.detach(), D_graph.detach()