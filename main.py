'''
Author: Gagandeep Randhawa
GaTech ID: grandhawa6
Course: CS8903

BSD 3-Clause License
Copyright (c) 2023, Gagandeep Randhawa
All rights reserved.

https://github.com/GaganCodes/CS8903-Spring22/blob/main/LICENSE

Code organization (separated by #++++++++++ and checkpoints):
1. Importing libraries (non-exhaustive)
2. Importing the dataset, separating into training and validation set
3. Defining relevant model and train/test step
4. Running the optimization loop
5. Testing the trained model with hold-out set
6. Saving the relevant results
'''

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Importing necessary libraries
import torch
from torch.utils.data import DataLoader
import numpy as np
from data_func import split_data_np, np_to_dataset, scale_features
from model_func import NeuralNetwork_1, NeuralNetwork_2
from epoch_func import train_loop, test_loop, check_early_stop
from loss_func import log_cosh_loss, eval_accuracy_max_abs_err
from post_func import plot_data, save_results_txt, plot_data_logx
import time

print("Checkpoint #1 - Libraries imported.")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Importing the dataset and defining other global parameters

t1 = time.time()
# Setting the random seed globally
torch.manual_seed(23)

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Hyperparameters
learning_rate = 1e-3
batch_size = 512

epochs = 500                # Min epochs to be at least 20 for consistent early stopping
min_epochs = int(epochs/4)  # Used for check_early_stop, using learning rate scheduler instead

loss_fn = log_cosh_loss
acc_fn = eval_accuracy_max_abs_err
loss_metric = "Log Cosh Loss"
acc_metric = "Max Abs Perct Error"

# Loading txt data into numpy arrays
all_data = np.loadtxt(".\\data\\all_data.txt")
train_size = int(0.70*len(all_data))
val_size = int(0.15*len(all_data))
test_size = len(all_data) - train_size - val_size
train_X, train_y, val_X, val_y, test_X, test_y = split_data_np(all_data[:,:4], all_data[:,-1], train_size, val_size, test_size, scale=True)

# Converting from numpy sets to torch dataloaders
train_dataloader = DataLoader(np_to_dataset(train_X, train_y), batch_size=batch_size)
val_dataloader = DataLoader(np_to_dataset(val_X, val_y), batch_size=batch_size)
test_dataloader = DataLoader(np_to_dataset(test_X, test_y), batch_size=batch_size)

print("Checkpoint #2 - Dataset imported.")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Defining NN model and train/test step
model = NeuralNetwork_2(4, 16, 16).to(device)
print(model)

optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate, eps=1e-07)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1, verbose=True)

print("Checkpoint #3 - Model and optimizer defined.")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Running optimization

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, acc_fn, device)
    test_loop(val_dataloader, model, loss_fn, acc_fn, device)
    scheduler.step()
    
test_loop(test_dataloader, model, loss_fn, acc_fn, device, mode='test')

plot_data_logx(np.asarray(model.train_score_metric_1), np.asarray(model.val_score_metric_1), loss_metric, model.name)

plot_data(np.asarray(model.train_score_metric_2), np.asarray(model.val_score_metric_2), acc_metric, model.name, scale=True)

save_results_txt(model, loss_metric, acc_metric)

t2 = time.time()
print(f"Runtime: {t2-t1}")
print("Checkpoint #4 - Results saved.")


