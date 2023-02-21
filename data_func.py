"""
Author: Gagandeep Randhawa
GaTech ID: grandhawa6
Course: CS8903

BSD 3-Clause License
Copyright (c) 2023, Gagandeep Randhawa
All rights reserved.

https://github.com/GaganCodes/CS8903-Spring22/blob/main/LICENSE

================
File with functions/classes relevant to pre-processing of the data
================
"""


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classes/functions specific to dataset
class load_data(Dataset):
    def __init__(self, data_file, device="cpu") -> None:
        """
        Class for PyTorch custom datset. This assumes training data is all in one file, with labels as the last column
        Ref: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        ================
        Parameters:
        data_file: name of the txt file in the .\data directory
        device: CPU or CUDA
        ================
        The .float() is used for resolving discrepancy of numpy's float64 to torch's float32 conversion,
        which becomes relevant later in using CUDA and computation graph.
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

class np_to_dataset(Dataset):
    def __init__(self, X, y) -> None:
        """
        Class for converting a numpy array to PyTorch custom datset.

        ** This additional class was defined to add scaling functionality in the code **

        Ref: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        ================
        Parameters:
        X: Numpy array of shape (Number of samples, Number of features)
        Y: Numpy array of shape (Number of samples, Number of classes) for classification problem
                                (Number of samples, ) for regression problem
        ================
        The .float() is used for resolving discrepancy of numpy's float64 to torch's float32 conversion,
        which becomes relevant later in using CUDA and computation graph.
        """

        self.X_torch = torch.from_numpy(X).float()
        self.Y_torch = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.Y_torch)

    def __getitem__(self, index):
        X = self.X_torch[index]
        y = self.Y_torch[index]

        return X, y


def scale_features(X_train, X_val, X_test):
    """
    Function for scaling the training features between [-1, 1] with zero mean and unit variance.
    Mean and Variance is calculated on the training set, then validation and test set are scaled accordingly.
    Ref:
        1. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        2. https://scikit-learn.org/stable/common_pitfalls.html#inconsistent-preprocessing
    ================
    Parameters:
    X_train, X_val, X_test: Numpy arrays of shape (Number of samples, Number of features)
    ================
    Returns:
    X_train_scaled, X_val_scaled, X_test_scaled: Scaled numpy arrays of shape (Number of samples, Number of features)
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled


def split_data_np(X, y, train_size=0.70, val_size=0.15, test_size=0.15, scale=False):
    """
    Function for splitting the data into training/validation/testing sets. Default ratios are float percentages,
    but can also be int length of the set (same as sklearn's inputs).
    ** This implementation assumes that there is always a train-val-test split. **
    Ref:
        1. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    ================
    Parameters:
    X: Numpy array of shape (Number of samples, Number of features)
    Y: Numpy array of shape (Number of samples, Number of classes) for classification problem
                            (Number of samples, ) for regression problem
    train_size, val_size, test_size: Float (% of the set) or length of dataset (<len(dataset))
    scale: Boolean, for scaling the features
    ================
    Returns:
    train_X, val_X, test_X: Numpy arrays of shape (Number of samples, Number of features)
    train_y, val_y, test_y: Numpy arrays of shape (Number of samples, Number of classes) for classification problem
                                                    (Number of samples, ) for regression problem
    """
    # Splitting first into test and (train+val), then splitting again in train, val
    # random_state=23 to stay consistent with torch manual seed
    train_val_X, test_X, train_val_y, test_y = train_test_split(X, y, test_size=test_size, random_state=23)
    train_X, val_X, train_y, val_y = train_test_split(train_val_X, train_val_y, test_size=val_size, random_state=23)

    if scale:
        train_X, val_X, test_X = scale_features(train_X, val_X, test_X)

    return train_X, train_y, val_X, val_y, test_X, test_y

    
def split_data(data, train_size=0.70, val_size=0.15, test_size=0.15, batch_size=256):
    """
    Function for splitting the torch dataset into training/validation/testing sets. Default ratios are float percentages,
    but can also be int length of the set (same as torch's inputs).
    ** This implementation assumes that there is always a train-val-test split. **
    Ref:
        1. https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
    ================
    Parameters:
    data: Custom dataset (Number of samples, Number of features)
    train_size, val_size, test_size: Float (% of the set) or length of dataset (<len(dataset))
    batch_size: Int, batch size used for optimizer
    ================
    Returns:
    train_dataloader, val_dataloader, test_dataloader: Dataloader class with shuffled data split
    """
    # generator seed = 23 to stay consistent with torch manual seed
    train_dataset, val_dataset, test_dataset = random_split(data,
                                                            [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(23))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader
