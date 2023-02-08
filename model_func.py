"""
Add description of this file
"""

import torch.nn as nn

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classes/functions specific to defining model
class NeuralNetwork_8_8(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork_8_8, self).__init__()
        self.linear_relu_stack = nn.Sequential(
                                            nn.Linear(4, 8),
                                            nn.ReLU(),
                                            nn.Linear(8, 8),
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
        self.name = "4x8x8x1"
    
    def forward(self, x):
        # Defining forward pass for the NN
        return self.linear_relu_stack(x)

class NeuralNetwork_16_16(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork_8_8, self).__init__()
        self.linear_relu_stack = nn.Sequential(
                                            nn.Linear(4, 16),
                                            nn.ReLU(),
                                            nn.Linear(16, 16),
                                            nn.ReLU(),
                                            nn.Linear(16, 1)
        )

        # For tracking metrics like loss, accuracy, etc for each phase
        # Kept two for now, can be modified
        self.train_score_metric_1 = list()
        self.train_score_metric_2 = list()
        self.val_score_metric_1 = list()
        self.val_score_metric_2 = list()
        self.test_score_metric_1 = list()
        self.test_score_metric_2 = list()
        self.name = "4x16x16x1"
    
    def forward(self, x):
        # Defining forward pass for the NN
        return self.linear_relu_stack(x)

class NeuralNetwork_16(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork_8_8, self).__init__()
        self.linear_relu_stack = nn.Sequential(
                                            nn.Linear(4, 16),
                                            nn.ReLU(),
                                            nn.Linear(16, 1)
        )

        # For tracking metrics like loss, accuracy, etc for each phase
        # Kept two for now, can be modified
        self.train_score_metric_1 = list()
        self.train_score_metric_2 = list()
        self.val_score_metric_1 = list()
        self.val_score_metric_2 = list()
        self.test_score_metric_1 = list()
        self.test_score_metric_2 = list()
        self.name = "4x16x1"
    
    def forward(self, x):
        # Defining forward pass for the NN
        return self.linear_relu_stack(x)

class NeuralNetwork_32(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork_8_8, self).__init__()
        self.linear_relu_stack = nn.Sequential(
                                            nn.Linear(4, 32),
                                            nn.ReLU(),
                                            nn.Linear(32, 1)
        )

        # For tracking metrics like loss, accuracy, etc for each phase
        # Kept two for now, can be modified
        self.train_score_metric_1 = list()
        self.train_score_metric_2 = list()
        self.val_score_metric_1 = list()
        self.val_score_metric_2 = list()
        self.test_score_metric_1 = list()
        self.test_score_metric_2 = list()
        self.name = "4x32x1"
    
    def forward(self, x):
        # Defining forward pass for the NN
        return self.linear_relu_stack(x)

class NeuralNetwork_64(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork_8_8, self).__init__()
        self.linear_relu_stack = nn.Sequential(
                                            nn.Linear(4, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, 1)
        )

        # For tracking metrics like loss, accuracy, etc for each phase
        # Kept two for now, can be modified
        self.train_score_metric_1 = list()
        self.train_score_metric_2 = list()
        self.val_score_metric_1 = list()
        self.val_score_metric_2 = list()
        self.test_score_metric_1 = list()
        self.test_score_metric_2 = list()
        self.name = "4x64x1"
    
    def forward(self, x):
        # Defining forward pass for the NN
        return self.linear_relu_stack(x)

