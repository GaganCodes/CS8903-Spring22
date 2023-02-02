'''
Name: Gagandeep Singh Randhawa
GTech User ID: grandhawa6
Course: CS8903

Code organization (separated by #++++++++++ and checkpoints):
1. Importing libraries (non-exhaustive)
2. Importing the dataset, separating into training and validation set
3. Defining relevant model and train/test step
4. Running the optimization loop
5. Testing the trained model with hold-out set
6. Saving the relevant information and figures
'''

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Importing necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import numpy as np
from util_classes import load_data
from util_classes import NeuralNetwork
from util_classes import train_loop, test_loop
from util_classes import log_cosh_loss, mean_L1_loss, max_abs_percent_err, eval_accuracy
from util_classes import plot_loss_data, plot_acc_data
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
batch_size = 1024
epochs = 150
loss_fn = mean_L1_loss
acc_fn = eval_accuracy
#metric = "Log Cosh Loss"
metric = "Mean Abs Perct Error"
#metric = "Max Abs Perct Err"

# Importing data and splitting in train, test, validate
all_data = load_data('all_data')
train_size = int(0.75*len(all_data))
test_size = int(0.15*len(all_data))
val_size = len(all_data) - train_size - test_size

train_dataset, test_dataset, val_dataset = random_split(all_data, [train_size, test_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

print("Checkpoint #2 - Dataset imported.")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Defining NN model and train/test step
model = NeuralNetwork()
print(model)

'''
# This was done for debugging dtype discrepancy btw NN and data
train_sample_batch, label_sample_batch = next(iter(train_dataloader))
train_sample_single, label_sample_single = train_sample_batch[0], label_sample_batch[0]
print(f"Datatype of training data: {train_sample_single.dtype}")

for param in model.parameters():
    print(type(param), param.dtype)
'''

print("Checkpoint #3 - Model and steps defined.")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Running optimization

optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate, eps=1e-07)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------") 
    train_loop(train_dataloader, model, loss_fn, optimizer, acc_fn)
    test_loop(val_dataloader, model, loss_fn, acc_fn)

test_loop(test_dataloader, model, loss_fn, acc_fn, mode='test')

plot_loss_data(np.asarray(model.train_score_metric_1), np.asarray(model.val_score_metric_1), epochs, metric)

plot_acc_data(np.asarray(model.train_score_metric_2), np.asarray(model.val_score_metric_2), epochs, "Accuracy")

t2 = time.time()
print(f"Runtime: {t2-t1}")
print("Checkpoint #4 - Figure saved.")
