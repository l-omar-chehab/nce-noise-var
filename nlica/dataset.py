"""Synthetic or Classic (e.g. MNIST, CIFAR) data for Pytorch usage."""

import torch
from torch.utils.data import Dataset


# Pytorch datasets


class SimpleDataset(Dataset):
    """Generic Dataset Class holding a pair (x, label).
    Args:
        X (torch.Tensor): observations, shape (n_samples, n_components)
        y (torch.Tensor): label, shape (n_samples, n_components=1)
        device (XXX): cpu or gpu
    """

    def __init__(self, X, y, device="cpu"):
        self.device = device
        dtype = torch.get_default_dtype()
        # self.x = torch.from_numpy(X).float().to(device)
        # self.y = torch.from_numpy(Y).float().to(device)
        self.X = X.type(dtype).to(device)
        self.y = y.type(dtype).to(device)
        self.len = self.X.shape[0]
        self.data_dim = self.X.shape[1]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class PairsDataset(Dataset):
    """Generic Dataset Class holding a pair (x, u, label).
    Args:
        X (torch.Tensor): observations, shape (n_samples, n_components)
        U (torch.Tensor): auxiliary variable, shape (n_samples, n_components)
                          (e.g. past, or time index, or transformation of x)
        y (torch.Tensor): label, shape (n_samples, n_components=1)
        device (XXX): cpu or gpu
    """

    def __init__(self, X, U, y, device="cpu"):
        self.device = device
        dtype = torch.get_default_dtype()
        self.X = X.type(dtype).to(device)
        self.U = U.type(dtype).to(device)
        self.y = y.type(dtype).to(device)
        self.len = self.X.shape[0]
        self.data_dim = self.X.shape[1]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.X[index], self.U[index], self.y[index]


class TuplesDataset(Dataset):
    """Generic Dataset Class holding a tuplet (x, u, u_norm, label).
    Args:
        X (torch.Tensor): observations, shape (n_samples, n_components)
        U (torch.Tensor): auxiliary variable, shape (n_samples, n_components)
                          (e.g. past, or time index, or transformation of x)
        U_norm (torch.Tensor): normalization variables, shape (n_samples, n_variables, n_components)
        y (torch.Tensor): label, shape (n_samples, n_components=1)
        device (XXX): cpu or gpu
    """

    def __init__(self, X, U, U_norm, y, device="cpu"):
        self.device = device
        dtype = torch.get_default_dtype()
        self.X = X.type(dtype).to(device)
        self.U = U.type(dtype).to(device)
        self.U_norm = U_norm.float().to(device)
        self.y = y.type(dtype).to(device)
        self.len = self.X.shape[0]
        self.data_dim = self.X.shape[1]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.X[index], self.U[index], self.U_norm[index], self.y[index]
