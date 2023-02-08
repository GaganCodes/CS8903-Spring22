"""
Add description of this file
"""


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classes/functions specific to dataset
class load_data(Dataset):
    def __init__(self, data_file, device="cpu") -> None:
        """
        This assumes training data is all in one file, with labels as the last column
        param data_file: name of the txt file in the .\data directory
        """
        self.all_data = np.loadtxt(f".\\data\\{data_file}.txt")

        self.X_torch = torch.from_numpy(self.all_data[:, 0:4]).float()
        self.Y_torch = torch.from_numpy(self.all_data[:, -1]).float()

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        X = self.X_torch[index]
        y = self.Y_torch[index]

        return X, y
    
def split_data(data, train_size=0.70, val_size=0.15, test_size=0.15, batch_size=256):
    train_dataset, val_dataset, test_dataset = random_split(data,
                                                            [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(23))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


class TensorStandardScaler:
    def __init__(self) -> None:
        self.mean = 0.0
        self.std = 0.0
    
    def fit(self, x: torch.tensor):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
    
    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-07)

        return x
