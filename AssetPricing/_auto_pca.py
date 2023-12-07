import numpy as np
import pandas as pd
import gc
from copy import copy
from tqdm import trange, tqdm
from datetime import datetime
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR


class StockDataset(Dataset):

    def __init__(self, stock_features, stock_returns):
        
        self.dates = list(stock_features.keys())
        self.stock_features = stock_features
        self.stock_returns = stock_returns

    def __len__(self):

        return len(self.dates)
    
    def __getitem__(self, index):

        month = self.dates[index]
        features = self.stock_features[month]
        returns = self.stock_returns.loc[month, self.stock_features[month].index].values

        feature_tensor = torch.tensor(features.values, dtype=torch.float32)
        return_tensor = torch.tensor(returns, dtype = torch.float32)

        return feature_tensor, return_tensor
    
def ols_residuals(X,Y):

    beta = torch.inverse(X.t().mm(X)).mm(X.t()).mm(Y)

    Y_pred = X.mm(beta)

    residuals = Y - Y_pred

    rss = torch.mean(residuals**2)

    return rss

def custom_collate(batch):
    features, labels = zip(*batch)
    return features, labels

def negative_correlation_loss(y_pred, y_true):
    # Calculate means
    y_pred_mean = torch.mean(y_pred)
    y_true_mean = torch.mean(y_true)

    # Numerator: covariance between predicted and true values
    cov = torch.mean((y_pred - y_pred_mean) * (y_true - y_true_mean))

    # Denominators: standard deviations of predicted and true values
    std_pred = torch.sqrt(torch.mean((y_pred - y_pred_mean) ** 2) + 1e-8)
    std_true = torch.sqrt(torch.mean((y_true - y_true_mean) ** 2) + 1e-8)

    # Correlation coefficient
    corr = cov / (std_pred * std_true)

    # Return negative correlation as loss
    return -corr


class Auto_PCA(nn.Module):
    def __init__(self, layer_list, sparsity_strength=0.01, sparse_layer = 0):
        super(Auto_PCA, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layer_list[i], layer_list[i+1]) for i in range(len(layer_list) - 1)
        ])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sparsity_strength = sparsity_strength
        self.first_layer_activation = None
        self.sparse_layer = sparse_layer

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == self.sparse_layer:  # Capture the activation of the first layer
                self.first_layer_activation = x
            x = self.relu(x)
        return x

    def sparsity_penalty(self):
        # Apply L1 penalty to encourage sparsity in the first layer
        return self.sparsity_strength * torch.mean(torch.abs(self.first_layer_activation))


