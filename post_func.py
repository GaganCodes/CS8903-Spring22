"""
Add description of this file
"""
import numpy as np
import matplotlib.pyplot as plt
import os

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classes/functions specific to visualizing relevant quantities
def plot_loss_data(train, val, metric, model_name):
    # Added parameter for naming files accordingly
    fig_iter = 0

    fig, ax = plt.subplots(figsize=(14,7))

    ax.plot(train, color='b', label="Training Set")
    ax.plot(val, color='r', label="Validation Set")

    ax.set_xlabel(f"# of Epochs")
    ax.set_ylabel(f"{metric}")

    ax.grid()
    ax.legend()

    ax.set_xlim(0.0, max(len(train), len(val)))
    ax.set_ylim(0.0)

    ax.set_title(f"{model_name}-Min Loss Test: {np.min(train):>0.2f}, Min Loss Val: {np.min(val):>0.2f}")

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

def plot_acc_data(train, val, acc_metric, model_name):
    # Added parameter for naming files accordingly
    fig_iter = 0

    fig, ax = plt.subplots(figsize=(14,7))

    ax.plot(train, color='b', label="Training Set")
    ax.plot(val, color='r', label="Validation Set")

    ax.set_xlabel(f"# of Epochs")
    ax.set_ylabel(f"{acc_metric} %")

    ax.grid()
    ax.legend()

    ax.set_xlim(0.0, max(len(train), len(val)))
    ax.set_ylim(0.0)

    ax.set_title(f"{model_name}-Min Acc Test: {np.min(train):>0.1f}%, Min Acc Val: {np.min(val):>0.1f}%")

    fig_path = f".\\Fig\\Fig-NN-{model_name}-{acc_metric}-{fig_iter}.png"    

    if not os.path.isfile(fig_path):
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")
    else:
        fig_iter += 1
        fig_path = f".\\Fig\\Fig-NN-{model_name}-{acc_metric}-{fig_iter}.png"
        while os.path.isfile(fig_path):
            fig_iter += 1
            fig_path = f".\\Fig\\Fig-NN-{model_name}-{acc_metric}-{fig_iter}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")

def plot_loss_data_logx(train, val, metric, model_name):
    # Added parameter for naming files accordingly
    fig_iter = 0

    fig, ax = plt.subplots(figsize=(14,7))

    ax.semilogx(train, color='b', label="Training Set")
    ax.semilogx(val, color='r', label="Validation Set")

    ax.set_xlabel(f"# of Epochs")
    ax.set_ylabel(f"{metric}")

    ax.grid()
    ax.legend()

    #ax.set_xlim(0.0, max(len(train), len(val)))
    ax.set_ylim(0.0)

    ax.set_title(f"{model_name}-Min Loss Test: {np.min(train):>0.2f}, Min Loss Val: {np.min(val):>0.2f}")

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

def plot_acc_data_logx(train, val, acc_metric, model_name):
    # Added parameter for naming files accordingly
    fig_iter = 0

    fig, ax = plt.subplots(figsize=(14,7))

    ax.semilogx(train, color='b', label="Training Set")
    ax.semilogx(val, color='r', label="Validation Set")

    ax.set_xlabel(f"# of Epochs")
    ax.set_ylabel(f"{acc_metric} %")

    ax.grid()
    ax.legend()

    #ax.set_xlim(0.0, max(len(train), len(val)))
    ax.set_ylim(0.0)

    ax.set_title(f"{model_name}-Min Acc Test: {np.min(train):>0.1f}%, Min Acc Val: {np.min(val):>0.1f}%")

    fig_path = f".\\Fig\\Fig-NN-{model_name}-{acc_metric}-{fig_iter}.png"    

    if not os.path.isfile(fig_path):
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")
    else:
        fig_iter += 1
        fig_path = f".\\Fig\\Fig-NN-{model_name}-{acc_metric}-{fig_iter}.png"
        while os.path.isfile(fig_path):
            fig_iter += 1
            fig_path = f".\\Fig\\Fig-NN-{model_name}-{acc_metric}-{fig_iter}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")

def save_results_txt(model, loss_metric=None, acc_metric=None):
    # Calculating relevant quantities
    # This case we have to store minimum value of loss
    # and minimum value of accuracy (Max abs percent error)
    train_metric_1 = np.min(model.train_score_metric_1)
    train_metric_2 = np.min(model.train_score_metric_2)
    val_metric_1 = np.min(model.val_score_metric_1)
    val_metric_2 = np.min(model.val_score_metric_2)
    if model.test_score_metric_1 and model.test_score_metric_2:
        test_metric_1 = np.min(model.test_score_metric_1)
        test_metric_2 = np.min(model.test_score_metric_2)

    file_iter = 0
    file_path = f".\\Results\\Results-NN-{model.name}-{file_iter}.txt"
    log_file_path = f".\\Results\\MAPE Log-NN-{model.name}-{file_iter}.txt"

    if not os.path.isfile(file_path):
        # Writing these quantities to the file
        output_file = open(file_path, 'w')
        output_file.write(model.name + '\n')
        output_file.write(f"Min train {loss_metric}: " + str(train_metric_1) + "\n")
        output_file.write(f"Min val {loss_metric}: " + str(val_metric_1) + "\n")
        if model.test_score_metric_1:
            output_file.write(f"Min test {loss_metric}: " + str(test_metric_1) + "\n")
        output_file.write(32*"+"+"\n")
        output_file.write(f"Min train {acc_metric}: " + str(train_metric_2) + "\n")
        output_file.write(f"Min val {acc_metric}: " + str(val_metric_2) + "\n")
        if model.test_score_metric_2:
            output_file.write(f"Min test {acc_metric}: " + str(test_metric_2) + "\n")
        output_file.write(32*"+"+"\n")
        output_file.close()

        # Writing training and validation history
        if acc_metric:
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
    else:
        file_iter += 1
        file_path = f".\\Results\\Results-NN-{model.name}-{file_iter}.txt"  
        log_file_path = f".\\Results\\MAPE Log-NN-{model.name}-{file_iter}.txt"
        while os.path.isfile(file_path):
            file_iter += 1
            file_path = f".\\Results\\Results-NN-{model.name}-{file_iter}.txt"  
            log_file_path = f".\\Results\\MAPE Log-NN-{model.name}-{file_iter}.txt"
        # Writing these quantities to the file
        output_file = open(file_path, 'w')
        output_file.write(model.name + '\n')
        output_file.write(f"Min train {loss_metric}: " + str(train_metric_1) + "\n")
        output_file.write(f"Min val {loss_metric}: " + str(val_metric_1) + "\n")
        if model.test_score_metric_1:
            output_file.write(f"Min test {loss_metric}: " + str(test_metric_1) + "\n")
        output_file.write(32*"+"+"\n")
        output_file.write(f"Min train {acc_metric}: " + str(train_metric_2) + "\n")
        output_file.write(f"Min val {acc_metric}: " + str(val_metric_2) + "\n")
        if model.test_score_metric_2:
            output_file.write(f"Min test {acc_metric}: " + str(test_metric_2) + "\n")
        output_file.write(32*"+"+"\n")
        output_file.close()

        # Writing training and validation history
        if acc_metric:
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
