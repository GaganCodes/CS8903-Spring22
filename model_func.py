"""
Author: Gagandeep Randhawa
GaTech ID: grandhawa6
Course: CS8903

BSD 3-Clause License
Copyright (c) 2023, Gagandeep Randhawa
All rights reserved.

https://github.com/GaganCodes/CS8903-Spring22/blob/main/LICENSE

================
File with functions/classes relevant to pre-defined neural network structures
================
"""

import torch.nn as nn

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classes/functions specific to defining model
class NeuralNetwork_1(nn.Module):
    '''
    Neural network with one hidden layer and ReLU activation layer.
    '''
    def __init__(self, num_input=4, num_hidden=8) -> None:
        super(NeuralNetwork_1, self).__init__()
        self.layer_1 = nn.Linear(num_input, num_hidden)
        self.activation_1 = nn.ReLU()
        self.output = nn.Linear(num_hidden,1)

        # For tracking metrics like loss, accuracy, etc for each phase
        # Kept two for now, can be modified
        self.train_score_metric_1 = list()
        self.train_score_metric_2 = list()
        self.val_score_metric_1 = list()
        self.val_score_metric_2 = list()
        self.test_score_metric_1 = list()
        self.test_score_metric_2 = list()
        self.name = "{}x{}x1".format(num_input, num_hidden)
    
    def forward(self, x):
        # Defining forward pass for the NN
        # Can be done in a line with torch.nn.Sequential
        # Done this way for readability
        z1 = self.layer_1(x)
        a1 = self.activation_1(z1)
        z2 = self.output(a1)
        return z2

class NeuralNetwork_2(nn.Module):
    '''
    Neural network with two hidden layers and ReLU activation layers.
    '''
    def __init__(self, num_input=4, num_hidden_1=8, num_hidden_2=8) -> None:
        super(NeuralNetwork_2, self).__init__()
        self.layer_1 = nn.Linear(num_input, num_hidden_1)
        self.activation_1 = nn.ReLU()
        self.layer_2 = nn.Linear(num_hidden_1, num_hidden_2)
        self.activation_2 = nn.ReLU()
        self.output = nn.Linear(num_hidden_2, 1)

        # For tracking metrics like loss, accuracy, etc for each phase
        # Kept two for now, can be modified
        self.train_score_metric_1 = list()
        self.train_score_metric_2 = list()
        self.val_score_metric_1 = list()
        self.val_score_metric_2 = list()
        self.test_score_metric_1 = list()
        self.test_score_metric_2 = list()
        self.name = "{}x{}x{}x1".format(num_input, num_hidden_1, num_hidden_2)
    
    def forward(self, x):
        # Defining forward pass for the NN
        # Can be done in a line with torch.nn.Sequential
        # Done this way for readability
        z1 = self.layer_1(x)
        a1 = self.activation_1(z1)
        z2 = self.layer_2(a1)
        a2 = self.activation_2(z2)
        z3 = self.output(a2)
        return z3
