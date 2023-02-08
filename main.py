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
import numpy as np
from data_func import load_data, split_data
from model_func import NeuralNetwork_8_8
from epoch_func import train_loop, test_loop, check_early_stop
from loss_func import log_cosh_loss, eval_accuracy_max_abs_err, log_cosh_std
from post_func import plot_loss_data, plot_acc_data, save_results_txt, plot_acc_data_logx, plot_loss_data_logx
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

# Min epochs to be at least 20 for consistent early stopping
epochs = 2000
min_epochs = int(epochs/1)

conv_criteria = 0.0001
loss_fn = log_cosh_loss
acc_fn = eval_accuracy_max_abs_err
loss_metric = "Log Cosh Loss"
acc_metric = "Max Abs Perct Error"

# Importing data and splitting in train, test, validate
all_data = load_data('all_data', device)
train_size = int(0.70*len(all_data))
#train_size = int(0.05*len(all_data))
val_size = int(0.15*len(all_data))
#val_size = int(0.01*len(all_data))
test_size = len(all_data) - train_size - val_size

train_dataloader, val_dataloader, test_dataloader = split_data(all_data, train_size, test_size, val_size, batch_size)

print(f"Train size: {len(train_dataloader.dataset)}")
print(f"Test size: {len(test_dataloader.dataset)}")
print(f"Val size: {len(val_dataloader.dataset)}")

print("Checkpoint #2 - Dataset imported.")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Defining NN model and train/test step
model = NeuralNetwork_8_8()

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
    train_loop(train_dataloader, model, loss_fn, optimizer, acc_fn, device)
    test_loop(val_dataloader, model, loss_fn, acc_fn, device)

    if t+1 > min_epochs:
        if check_early_stop(model):
            print(f"Stopping early: Validation loss is increasing.")
            break
        else:
            continue

    
test_loop(test_dataloader, model, loss_fn, acc_fn, mode='test')

plot_loss_data(np.asarray(model.train_score_metric_1), np.asarray(model.val_score_metric_1), loss_metric, model.name)

plot_acc_data(np.asarray(model.train_score_metric_2), np.asarray(model.val_score_metric_2), acc_metric, model.name)

save_results_txt(model, loss_metric, acc_metric)

t2 = time.time()
print(f"Runtime: {t2-t1}")
print("Checkpoint #4 - Results saved.")
