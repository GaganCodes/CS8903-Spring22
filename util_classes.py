"""
Add description of this file
"""


import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler

class load_data(Dataset):
    def __init__(self, data_file, scale=False) -> None:
        """
        This assumes training data is all in one file only, with labels as the last column
        param data_file: name of the txt file in the .\data directory
        """
        self.all_data = np.loadtxt(f".\\data\\{data_file}.txt")

        if scale:
            X = StandardScaler().fit_transform(self.all_data[:, 0:4])
            self.X_torch = torch.from_numpy(X).float()
        else:
            self.X_torch = torch.from_numpy(self.all_data[:, 0:4]).float()

        self.Y_torch = torch.from_numpy(self.all_data[:, -1]).float()

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        X = self.X_torch[index]
        y = self.Y_torch[index]

        return X, y


def plot_data(mape_train, mape_val, epochs):
    fig, ax = plt.subplots(figsize=(14,7))
    ax.grid()
    ax.plot(mape_train, color='b', label="Training Set")
    ax.plot(mape_val, color='r', label="Validation Set")
    ax.set_xlabel("# of Epochs")
    ax.set_ylabel("Max. APE %")
    ax.legend()
    ax.set_xlim(0.0)
    ax.set_ylim(0.0)
    ax.set_title(f"Min MAPE Train: {np.min(mape_train):>0.1f}%, Min MAPE Val: {np.min(mape_val):>0.1f}%")
    plt.savefig(f".\\Fig\\Fig-01-NN-MAPE vs Epochs-{epochs}.png",
        bbox_inches='tight', dpi=400, format="png")
