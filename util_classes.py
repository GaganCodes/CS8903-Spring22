"""
Add description of this file
"""


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classes/functions specific to dataset
class load_data(Dataset):
    def __init__(self, data_file, scale=False) -> None:
        """
        This assumes training data is all in one file, with labels as the last column
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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classes/functions specific to defining model

# NN with 2 hidden layers and ReLU activation between layers
class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                                            nn.Linear(4, 8),
                                            nn.ReLU(),
                                            nn.Linear(8, 1)
        )

        # For tracking metrics like loss, accuracy, etc for each phase
        # Kept two for now, can be modified
        self.train_score_metric_1 = list()
        self.train_score_metric_2 = list()
        self.val_score_metric_1 = list()
        self.val_score_metric_2 = list()
        self.test_score_metric_1 = list()
        self.test_score_metric_2 = list()

    
    def forward(self, x):
        # Defining forward pass for the NN
        return self.linear_relu_stack(x)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classes/functions specific to loss function
def mean_L1_loss(y_pred=None, y_true=None):
    loss_fn = nn.L1Loss()
    # Assuming y_pred and y_true are tensors
    # Following check ensures there is no mismatch in tensor dimensions
    # which can lead to a square tensor of size nxn instead of column vector of size nx1
    # Assuming y_pred.size() == y_true.size() regardless of orientation
    if y_pred.dim() == y_true.dim():
        norm_pred = torch.div(y_pred, y_true)
        norm_y = torch.div(y_true, y_true)
    if y_pred.dim() > y_true.dim():
        # This implies y_pred is torch.size([n, 1]) and y_true is torch.size([n])
        norm_pred = torch.div(y_pred, torch.unsqueeze(y_true, 1))
        norm_y = torch.div(torch.unsqueeze(y_true, 1), torch.unsqueeze(y_true, 1))
    if y_pred.dim() < y_true.dim():
        # This implies y_pred is torch.size([n]) and y_true is torch.size([n, 1])
        norm_pred = torch.div(torch.unsqueeze(y_pred, 1), y_true)
        norm_y = torch.div(y_true, y_true)
    
    return loss_fn(norm_pred, norm_y)

def log_cosh_loss(y_pred=None, y_true=None):
    # Assuming y_pred and y_true are tensors
    # Following check ensures there is no mismatch in tensor dimensions
    # which can lead to a square tensor of size nxn instead of column vector of size nx1
    # Assuming y_pred.size() == y_true.size() regardless of orientation
    if y_pred.dim() == y_true.dim():
        cosh_term = torch.subtract(y_pred, y_true)
        log_term = torch.cosh(cosh_term)
    if y_pred.dim() > y_true.dim():
        # This implies y_pred is torch.size([n, 1]) and y_true is torch.size([n])
        cosh_term = torch.subtract(y_pred, torch.unsqueeze(y_true, 1))
        log_term = torch.cosh(cosh_term)
    if y_pred.dim() < y_true.dim():
        # This implies y_pred is torch.size([n]) and y_true is torch.size([n, 1])
        cosh_term = torch.subtract(torch.unsqueeze(y_pred, 1), y_true)
        log_term = torch.cosh(cosh_term)

    return torch.mean(torch.log(log_term))

def max_abs_percent_err(y_pred=None, y_true=None):
    # Assuming y_pred and y_true are tensors
    # Following check ensures there is no mismatch in tensor dimensions
    # which can lead to a square tensor of size nxn instead of column vector of size nx1
    # Assuming y_pred.size() == y_true.size() regardless of orientation
    if y_pred.dim() == y_true.dim():
        norm_pred = torch.div(y_pred, y_true)
        norm_y = torch.div(y_true, y_true)
    if y_pred.dim() > y_true.dim():
        # This implies y_pred is torch.size([n, 1]) and y_true is torch.size([n])
        norm_pred = torch.div(y_pred, torch.unsqueeze(y_true, 1))
        norm_y = torch.div(torch.unsqueeze(y_true, 1), torch.unsqueeze(y_true, 1))
    if y_pred.dim() < y_true.dim():
        # This implies y_pred is torch.size([n]) and y_true is torch.size([n, 1])
        norm_pred = torch.div(torch.unsqueeze(y_pred, 1), y_true)
        norm_y = torch.div(y_true, y_true)
    
    return torch.max(torch.abs(torch.subtract(norm_pred, norm_y)))

def eval_accuracy(y_pred=None, y_true=None):
    # This function assumes a regression problem, and defines accuracy as being within 5% of the actual value
    # i.e., 0.95 <= |y_pred/y_true| <= 1.05

    # Assuming y_pred and y_true are tensors
    # Following check ensures there is no mismatch in tensor dimensions
    # which can lead to a square tensor of size nxn instead of column vector of size nx1
    # Assuming y_pred.size() == y_true.size() regardless of orientation
    if y_pred.dim() == y_true.dim():
        norm_pred = torch.div(y_pred, y_true)
        norm_y = torch.div(y_true, y_true)
    if y_pred.dim() > y_true.dim():
        # This implies y_pred is torch.size([n, 1]) and y_true is torch.size([n])
        norm_pred = torch.div(y_pred, torch.unsqueeze(y_true, 1))
        norm_y = torch.div(torch.unsqueeze(y_true, 1), torch.unsqueeze(y_true, 1))
    if y_pred.dim() < y_true.dim():
        # This implies y_pred is torch.size([n]) and y_true is torch.size([n, 1])
        norm_pred = torch.div(torch.unsqueeze(y_pred, 1), y_true)
        norm_y = torch.div(y_true, y_true)
    
    # Accuracy is sum of num of elements with normalized value <= 0.05
    return torch.sum(torch.lt(torch.abs(torch.subtract(norm_pred, norm_y)), 0.05).int()).item()/len(norm_pred)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classes/functions specific to epoch/prediction steps
def train_loop(dataloader, model, loss_fn, optimizer, acc_fn=None, record=True):
    num_batches = len(dataloader)
    train_loss = 0.0
    accuracy = 0.0

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)

        loss = loss_fn(pred, y)
        train_loss += loss.item()

        if acc_fn:
            accuracy += acc_fn(pred, y)
    
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= num_batches
    accuracy /= num_batches
    if record:
        model.train_score_metric_1.append(100*train_loss)
        if acc_fn:
            model.train_score_metric_2.append(100*accuracy)

def test_loop(dataloader, model, loss_fn, acc_fn=None, record=True, mode='val'):
    num_batches = len(dataloader)
    test_loss = 0.0
    accuracy = 0.0

    # Not keeping track of gradient because we're only testing = forward pass only
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            if acc_fn:
                accuracy += acc_fn(pred, y)
    
    test_loss /= num_batches
    accuracy /= num_batches
    print(f"Avg loss: {(100*test_loss):>0.1f}% \n")

    if record and mode=='val':
        model.val_score_metric_1.append(100*test_loss)
        if acc_fn:
            model.val_score_metric_2.append(100*accuracy)
    if record and mode=='test':
        model.test_score_metric_1.append(100*test_loss)
        if acc_fn:
            model.test_score_metric_2.append(100*accuracy)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classes/functions specific to visualizing relevant quantities
def plot_loss_data(train, val, epochs, metric):
    fig, ax = plt.subplots(figsize=(14,7))

    ax.plot(train, color='b', label="Training Set")
    ax.plot(val, color='r', label="Validation Set")

    ax.set_xlabel("# of Epochs")
    ax.set_ylabel(f"{metric} %")

    ax.grid()
    ax.legend()

    ax.set_xlim(0.0, epochs-1)
    ax.set_ylim(0.0)

    ax.set_title(f"Min Loss Test: {np.min(train):>0.1f}%, Min Loss Val: {np.min(val):>0.1f}%")

    fig_iter = 0
    fig_path = f".\\Fig\\Fig-NN-{metric} vs {epochs} Epochs-{fig_iter}.png"
    
    # Added parameter for naming files accordingly
    fig_name_add_appender = "-8x8-Unscaled"

    if not os.path.isfile(fig_path):
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")
    else:
        fig_iter += 1
        fig_path = f".\\Fig\\Fig-NN{fig_name_add_appender}-{metric} vs {epochs} Epochs-{fig_iter}.png"
        while os.path.isfile(fig_path):
            fig_iter += 1
            fig_path = f".\\Fig\\Fig-NN{fig_name_add_appender}-{metric} vs {epochs} Epochs-{fig_iter}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")

def plot_acc_data(train, val, epochs, metric):
    fig, ax = plt.subplots(figsize=(14,7))

    ax.plot(train, color='b', label="Training Set")
    ax.plot(val, color='r', label="Validation Set")

    ax.set_xlabel("# of Epochs")
    ax.set_ylabel(f"{metric} %")

    ax.grid()
    ax.legend()

    ax.set_xlim(0.0, epochs-1)
    ax.set_ylim(0.0)

    ax.set_title(f"Max Acc Test: {np.max(train):>0.1f}%, Max Acc Val: {np.max(val):>0.1f}%")

    fig_iter = 0
    fig_path = f".\\Fig\\Fig-NN-{metric} vs {epochs} Epochs-{fig_iter}.png"
    
    # Added parameter for naming files accordingly
    fig_name_add_appender = "-8x8-Unscaled"

    if not os.path.isfile(fig_path):
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")
    else:
        fig_iter += 1
        fig_path = f".\\Fig\\Fig-NN{fig_name_add_appender}-{metric} vs {epochs} Epochs-{fig_iter}.png"
        while os.path.isfile(fig_path):
            fig_iter += 1
            fig_path = f".\\Fig\\Fig-NN{fig_name_add_appender}-{metric} vs {epochs} Epochs-{fig_iter}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")
