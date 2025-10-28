# -*- coding:utf-8 -*-
"""
Filename: engine.py
Author: lisheng
Time: 2025-04-01
"""

import numpy as np
import os
import torch
import torch.nn as nn


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    for category in ["train", "val", "test"]:
        cat_data = np.load(os.path.join(dataset_dir, category + ".npz"))
        data["x_" + category] = cat_data["x"]
        data["y_" + category] = cat_data["y"]
    scaler = StandardScaler(
        mean=data["x_train"][..., 0].mean(), std=data["x_train"][..., 0].std()
    )
    # Data format
    for category in ["train", "val", "test"]:
        data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])


    print("Perform shuffle on the dataset")
    random_train = torch.arange(int(data["x_train"].shape[0]))
    random_train = torch.randperm(random_train.size(0))
    data["x_train"] = data["x_train"][random_train, ...]
    data["y_train"] = data["y_train"][random_train, ...]

    random_val = torch.arange(int(data["x_val"].shape[0]))
    random_val = torch.randperm(random_val.size(0))
    data["x_val"] = data["x_val"][random_val, ...]
    data["y_val"] = data["y_val"][random_val, ...]


    data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size)
    data["val_loader"] = DataLoader(data["x_val"], data["y_val"], valid_batch_size)
    data["test_loader"] = DataLoader(data["x_test"], data["y_test"], test_batch_size)
    data["scaler"] = scaler

    return data


def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        return torch.mean(torch.abs(true - pred))
    else:
        return torch.mean(torch.abs(true - pred))

def MAE_Freq_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        return torch.mean(torch.abs(torch.fft.rfft(pred, dim=-1) - torch.fft.rfft(true, dim=-1)))
    else:
        return torch.mean(torch.abs(torch.fft.rfft(pred, dim=-1) - torch.fft.rfft(true, dim=-1)))

def MAE_Time_Frequency_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        time_loss = torch.mean(torch.abs(true - pred))
        frequency_loss = torch.mean(torch.abs(torch.fft.rfft(pred, dim=-1) - torch.fft.rfft(true, dim=-1)))
        alpha = nn.Parameter(torch.tensor(-5.0))
        alpha_sigmoid = torch.sigmoid(alpha)
        loss = alpha_sigmoid * time_loss + frequency_loss * (1 - alpha_sigmoid * time_loss)
        return loss
    else:
        return torch.mean(torch.abs(true - pred)) + torch.mean(torch.abs(torch.fft.rfft(pred, dim=-1) - torch.fft.rfft(true, dim=-1)))

def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        return torch.mean(torch.abs(torch.div((true - pred), true)))
    else:
        return torch.mean(torch.abs(torch.div((true - pred), true)))


def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        return torch.sqrt(torch.mean((pred - true) ** 2))
    else:
        return torch.sqrt(torch.mean((pred - true) ** 2))


def WMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        loss = torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))
        return loss
    else:
        return torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))


def metric(pred, real):
    print('pred.shape :', pred.shape)
    print('real.shape :', real.shape)
    mae = MAE_torch(pred, real, 0.0).item()
    mape = MAPE_torch(pred, real, 0.0).item()
    wmape = WMAPE_torch(pred, real, 0.0).item()
    rmse = RMSE_torch(pred, real, 0.0).item()
    return mae, mape, rmse, wmape


def metric_per_station(pred, true, mask_value=None):
    mae_per_station = MAE_per_station_torch(pred, true, mask_value)
    mape_per_station = MAPE_per_station_torch(pred, true, mask_value)
    rmse_per_station = RMSE_per_station_torch(pred, true, mask_value)
    return mae_per_station.cpu().numpy(), mape_per_station.cpu().numpy(), rmse_per_station.cpu().numpy()


def MAE_per_station_torch(pred, true, mask_value=None):
    abs_error = torch.abs(true - pred)
    if mask_value is not None:
        mask = (true > mask_value).float()
        abs_error = abs_error * mask
        denom = mask.sum(dim=(0, 1, 3)) + 1e-6
    else:
        denom = torch.tensor(pred.shape[0] * pred.shape[1] * pred.shape[3], dtype=torch.float, device=pred.device)
    return abs_error.sum(dim=(0, 1, 3)) / denom


def RMSE_per_station_torch(pred, true, mask_value=None):
    sq_error = (true - pred) ** 2
    if mask_value is not None:
        mask = (true > mask_value).float()
        sq_error = sq_error * mask
        denom = mask.sum(dim=(0, 1, 3)) + 1e-6
    else:
        denom = torch.tensor(pred.shape[0] * pred.shape[1] * pred.shape[3], dtype=torch.float, device=pred.device)
    return torch.sqrt(sq_error.sum(dim=(0, 1, 3)) / denom)


def MAPE_per_station_torch(pred, true, mask_value=None):
    eps = 1e-6
    mape_error = torch.abs(pred - true) / (true.abs() + eps)
    if mask_value is not None:
        mask = (true > mask_value).float()
        mape_error = mape_error * mask
        denom = mask.sum(dim=(0, 1, 3)) + eps
    else:
        denom = torch.tensor(pred.shape[0] * pred.shape[1] * pred.shape[3], dtype=torch.float, device=pred.device)
    return mape_error.sum(dim=(0, 1, 3)) / denom




def metric_per_station_TPH(pred, true, mask_value=None):
    mae_per_station_TPH = MAE_per_station_TPH_torch(pred, true, mask_value)
    mape_per_station_TPH = MAPE_per_station_TPH_torch(pred, true, mask_value)
    rmse_per_station_TPH = RMSE_per_station_TPH_torch(pred, true, mask_value)
    return mae_per_station_TPH.cpu().numpy(), mape_per_station_TPH.cpu().numpy(), rmse_per_station_TPH.cpu().numpy()


def MAE_per_station_TPH_torch(pred, true, mask_value=None):
    abs_error = torch.abs(true - pred)
    if mask_value is not None:
        mask = (true > mask_value).float()
        abs_error = abs_error * mask
        denom = mask.sum(dim=(0, 1)) + 1e-6
    else:
        denom = torch.tensor(pred.shape[0] * pred.shape[1] * pred.shape[3], dtype=torch.float, device=pred.device)
    return abs_error.sum(dim=(0, 1)) / denom


def RMSE_per_station_TPH_torch(pred, true, mask_value=None):
    sq_error = (true - pred) ** 2
    if mask_value is not None:
        mask = (true > mask_value).float()
        sq_error = sq_error * mask
        denom = mask.sum(dim=(0, 1)) + 1e-6
    else:
        denom = torch.tensor(pred.shape[0] * pred.shape[1] * pred.shape[3], dtype=torch.float, device=pred.device)
    return torch.sqrt(sq_error.sum(dim=(0, 1)) / denom)


def MAPE_per_station_TPH_torch(pred, true, mask_value=None):
    eps = 1e-6
    mape_error = torch.abs(pred - true) / (true.abs() + eps)
    if mask_value is not None:
        mask = (true > mask_value).float()
        mape_error = mape_error * mask
        denom = mask.sum(dim=(0, 1)) + eps
    else:
        denom = torch.tensor(pred.shape[0] * pred.shape[1] * pred.shape[3], dtype=torch.float, device=pred.device)
    return mape_error.sum(dim=(0, 1)) / denom