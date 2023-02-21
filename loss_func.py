"""
Author: Gagandeep Randhawa
GaTech ID: grandhawa6
Course: CS8903

BSD 3-Clause License
Copyright (c) 2023, Gagandeep Randhawa
All rights reserved.

https://github.com/GaganCodes/CS8903-Spring22/blob/main/LICENSE

================
File with functions/classes relevant to calculating different loss/accuracy criteria
================
"""


import torch
import torch.nn as nn
import numpy as np

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classes/functions specific to loss function

def norm_pred_target(y_pred=None, y_true=None):
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

    return norm_pred, norm_y

def mean_L1_loss(y_pred=None, y_true=None):
    loss_fn = nn.L1Loss()
    norm_pred, norm_y = norm_pred_target(y_pred, y_true)    

    return loss_fn(norm_pred, norm_y)

def log_cosh_loss(y_pred=None, y_true=None):
    norm_pred, norm_y = norm_pred_target(y_pred, y_true)    
    cosh_tuning_parameter = 1.0
    cosh_term = torch.subtract(norm_pred, norm_y)
    log_term = torch.cosh(cosh_tuning_parameter * cosh_term)

    return torch.mean(torch.log(log_term))

def log_cosh_std(y_pred=None, y_true=None):
    norm_pred, norm_y = norm_pred_target(y_pred, y_true)    
    cosh_term = torch.subtract(norm_pred, norm_y)
    log_term = torch.cosh(cosh_term)

    return torch.mean(torch.log(log_term))

def mean_max_abs_percent_err(y_pred=None, y_true=None):
    norm_pred, norm_y = norm_pred_target(y_pred, y_true)    
    
    return torch.max(torch.abs(torch.subtract(norm_pred, norm_y)))

def eval_accuracy_rel_err(y_pred=None, y_true=None):
    norm_pred, norm_y = norm_pred_target(y_pred, y_true)    
    
    # Accuracy is sum of num of elements with normalized value <= 0.05
    return torch.sum(torch.lt(torch.abs(torch.subtract(norm_pred, norm_y)), 0.05).int()).item()/len(norm_pred)

def eval_accuracy_max_abs_err(y_pred=None, y_true=None):
    norm_pred, norm_y = norm_pred_target(y_pred, y_true)    
    
    # Accuracy is element with the maximum absolute relative error
    return torch.max(torch.abs(torch.subtract(norm_pred, norm_y)))

