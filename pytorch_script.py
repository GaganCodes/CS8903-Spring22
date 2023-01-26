'''
Name: Gagandeep Singh Randhawa
GTech User ID: grandhawa6
Course: CS8903

+++
Main training and testing code for Neural Network performance
TODO:
    1. Refactor away the model
    2. Refactor away the train and test functions
    3. Find a way to store the values
+++

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
from util_classes import load_data, plot_data
import time

print("Checkpoint #1 - Libraries imported.")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Importing the dataset and defining other global parameters

t1 = time.time()
# For readability in terminal
line_break = "\n"+64*"+"+"\n"

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Hyperparameters
learning_rate = 1e-3
batch_size = 1024
epochs = 200
loss_fn = nn.L1Loss()

# Importing data and splitting in train, test, validate
all_data = load_data('all_data')
train_size = int(0.75*len(all_data))
test_size = int(0.15*len(all_data))
val_size = len(all_data) - train_size - test_size

train_dataset, test_dataset, val_dataset = random_split(all_data, [train_size, test_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

'''
train_sample_batch, label_sample_batch = next(iter(train_dataloader))
train_sample_single, label_sample_single = train_sample_batch[0], label_sample_batch[0]
print(f"Datatype of training data: {train_sample_single.dtype}")
'''

# Tracking the MAPE
mape_train = list()
mape_val = list()

print("Checkpoint #2 - Dataset imported.")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Defining NN model and train/test step

# Defining model
class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                                            nn.Linear(4, 8),
                                            nn.ReLU(),
                                            nn.Linear(8, 8),
                                            nn.ReLU(),
                                            nn.Linear(8, 1),
        )
    
    def forward(self, x):
        # Defining forward pass for the NN
        return self.linear_relu_stack(x)

model = NeuralNetwork()

'''
for param in model.parameters():
    print(type(param), param.dtype)

'''
# Defining training and testing steps
def train_loop(dataloader, model, loss_fn, optimizer, record=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0.0

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)

        # Normalizing predicted value and y for MAPE
        norm_pred = torch.div(pred, torch.unsqueeze(y, 1))
        norm_y = torch.div(torch.unsqueeze(y, 1), torch.unsqueeze(y, 1))
        loss = loss_fn(norm_pred, norm_y)

        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''
        if batch%1000 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        '''
    train_loss /= num_batches
    if record:
        mape_train.append(100*train_loss)

def test_loop(dataloader, model, loss_fn, record=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0.0

    # Not keeping track of gradient because we're only testing = forward pass only
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            # Normalizing predicted value and y for MAPE
            norm_pred = torch.div(pred, torch.unsqueeze(y, 1))
            norm_y = torch.div(torch.unsqueeze(y, 1), torch.unsqueeze(y, 1))
            test_loss += loss_fn(norm_pred, norm_y).item()
    
    test_loss /= num_batches
    print(f"Avg loss: {(100*test_loss):>0.1f}% \n")

    if record:
        mape_val.append(100*test_loss)

print("Checkpoint #3 - Model and steps defined.")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Running optimization

optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)

for t in range(epochs):
    if (t+1)%100:
        print(f"Epoch {t+1}\n-------------------------------") 
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(val_dataloader, model, loss_fn)

test_loop(test_dataloader, model, loss_fn, record=False)

plot_data(np.asarray(mape_train), np.asarray(mape_val), epochs)

t2 = time.time()
print(f"Runtime: {t2-t1}")
print("Checkpoint #4 - Figure saved.")
