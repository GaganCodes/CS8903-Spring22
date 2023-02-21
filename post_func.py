"""
Author: Gagandeep Randhawa
GaTech ID: grandhawa6
Course: CS8903

BSD 3-Clause License
Copyright (c) 2023, Gagandeep Randhawa
All rights reserved.

https://github.com/GaganCodes/CS8903-Spring22/blob/main/LICENSE
================
File with functions/classes relevant to post-processing of data, example: plotting loss curves, accuracy curves, etc.
================
"""

import numpy as np
import matplotlib.pyplot as plt
import os

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classes/functions specific to visualizing relevant quantities
def plot_data(train, val, metric, model_name, scale=False):
    """
    Function for plotting the metric history vs epochs for a particular model.
    ================
    Parameters:
    train, val: Numpy arrays of metric value history (Number of epochs)
    metric: String, the metric being visualized (loss, accuracy, etc.)
    model_name: String, description of model architecture
    scale: Boolean, when True restricts Y-axis limits to (0, 0.20*Max metric value)
    ================
    Returns:
    Nothing, but saves the figure without overwriting the existing figures.
    """
    fig, ax = plt.subplots(figsize=(14,7))

    ax.plot(train, color='b', label="Training Set")
    ax.plot(val, color='r', label="Validation Set")

    ax.set_xlabel(f"# of Epochs")
    ax.set_ylabel(f"{metric}")

    ax.grid()
    ax.legend()

    ax.set_xlim(0.0, max(len(train), len(val)))
    if scale:
        ax.set_ylim(0.0, 0.20*max(np.max(train), np.max(val)))
    else:
        ax.set_ylim(0.0)

    ax.set_title(f"{model_name}-Min Test: {np.min(train):>0.2f}, Min Val: {np.min(val):>0.2f}")

    # Added parameter for naming files accordingly
    fig_iter = 0
    fig_path = f".\\Fig\\Fig-NN-{model_name}-{metric}-{fig_iter}.png"

    if not os.path.isfile(fig_path):
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")
    else:
        fig_iter += 1
        fig_path = f".\\Fig\\Fig-NN-{model_name}-{metric}-{fig_iter}.png"
        while os.path.isfile(fig_path):
            fig_iter += 1
            fig_path = f".\\Fig\\Fig-NN-{model_name}-{metric}-{fig_iter}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")


def plot_data_logx(train, val, metric, model_name, scale=False):
    """
    Function for plotting the metric history vs epochs for a particular model on a log-10 scale for X-axis.
    ================
    Parameters:
    train, val: Numpy arrays of metric value history (Number of epochs)
    metric: String, the metric being visualized (loss, accuracy, etc.)
    model_name: String, description of model architecture
    scale: Boolean, when True restricts Y-axis limits to (0, 0.20*Max metric value)
    ================
    Returns:
    Nothing, but saves the figure without overwriting the existing figures.
    """
    fig, ax = plt.subplots(figsize=(14,7))

    ax.semilogx(train, color='b', label="Training Set")
    ax.semilogx(val, color='r', label="Validation Set")

    ax.set_xlabel(f"# of Epochs")
    ax.set_ylabel(f"{metric}")

    ax.grid()
    ax.legend()

    if scale:
        ax.set_ylim(0.0, 0.20*max(np.max(train), np.max(val)))
    else:
        ax.set_ylim(0.0)

    ax.set_title(f"{model_name}-Min Test: {np.min(train):>0.2f}, Min Val: {np.min(val):>0.2f}")

    # Added parameter for naming files accordingly
    fig_iter = 0
    fig_path = f".\\Fig\\Fig-NN-{model_name}-{metric}-{fig_iter}.png"

    if not os.path.isfile(fig_path):
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")
    else:
        fig_iter += 1
        fig_path = f".\\Fig\\Fig-NN-{model_name}-{metric}-{fig_iter}.png"
        while os.path.isfile(fig_path):
            fig_iter += 1
            fig_path = f".\\Fig\\Fig-NN-{model_name}-{metric}-{fig_iter}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")


def save_results_txt(model, loss_metric=None, acc_metric=None):
    """
    Function for saving the relevant model run information. Not the same as torch's model save method.
    ================
    Parameters:
    train, val: Numpy arrays of metric value history (Number of epochs)
    metric: String, the metric being visualized (loss, accuracy, etc.)
    model_name: String, description of model architecture
    scale: Boolean, when True restricts Y-axis limits to (0, 0.20*Max metric value)
    ================
    Returns:
    Nothing, but saves the figure without overwriting the existing figures.
    Files saved:
        1. Results-NN-Model name: Txt file with minimum values for loss and accurary
        2. Metric Log-NN-Model name: Txt file with log for max abs perct error (train + val)
    """
    # Defining function for modularity
    def write_txt(model, file_path, log_file_path, loss_metric, acc_metric):
        # Writing these quantities to the file
        output_file = open(file_path, 'w')
        output_file.write(model.name + '\n')
        output_file.write(f"Min train {loss_metric}: " + str(np.min(model.train_score_metric_1)) + "\n")
        output_file.write(f"Min val {loss_metric}: " + str(np.min(model.val_score_metric_1)) + "\n")
        if model.test_score_metric_1:
            output_file.write(f"Min test {loss_metric}: " + str(np.min(model.test_score_metric_1)) + "\n")
        output_file.write(32*"+"+"\n")
        output_file.write(f"Min train {acc_metric}: " + str(np.min(model.train_score_metric_2)) + "\n")
        output_file.write(f"Min val {acc_metric}: " + str(np.min(model.val_score_metric_2)) + "\n")
        if model.test_score_metric_2:
            output_file.write(f"Min test {acc_metric}: " + str(np.min(model.test_score_metric_2)) + "\n")
        output_file.write(32*"+"+"\n")
        output_file.close()

        # Writing history
        output_file = open(log_file_path, 'w')
        output_file.write(model.name + '\n')
        output_file.write("Train Log: \n")
        for acc in model.train_score_metric_2:
            output_file.write(str(acc)+'\n')
        output_file.write('\n')
        output_file.write(32*"+"+"\n")
        output_file.write("Validation Log: \n")
        for acc in model.val_score_metric_2:
            output_file.write(str(acc)+'\n')
        output_file.close()


    file_iter = 0
    file_path = f".\\Results\\Results-NN-{model.name}-{file_iter}.txt"
    log_file_path = f".\\Results\\Metric Log-NN-{model.name}-{file_iter}.txt"

    if not os.path.isfile(file_path):
        write_txt(model, file_path, log_file_path, loss_metric, acc_metric)
    else:
        file_iter += 1
        file_path = f".\\Results\\Results-NN-{model.name}-{file_iter}.txt"  
        log_file_path = f".\\Results\\Metric Log-NN-{model.name}-{file_iter}.txt"
        while os.path.isfile(file_path):
            file_iter += 1
            file_path = f".\\Results\\Results-NN-{model.name}-{file_iter}.txt"  
            log_file_path = f".\\Results\\Metric Log-NN-{model.name}-{file_iter}.txt"

        write_txt(model, file_path, log_file_path, loss_metric, acc_metric)

