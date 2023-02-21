'''
Author: Gagandeep Randhawa
GaTech ID: grandhawa6
Course: CS8903

BSD 3-Clause License
Copyright (c) 2023, Gagandeep Randhawa
All rights reserved.

https://github.com/GaganCodes/CS8903-Spring22/blob/main/LICENSE

================
Additional code written to understand the feature and target value distribution.
================
'''

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Importing necessary libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# For plotting figures with large number of points
mpl.rcParams['agg.path.chunksize'] = 1000

print("Checkpoint #1 - Libraries imported.")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # Importing the dataset for this file
data = np.loadtxt(".\\data\\all_data.txt")

X_np = data[:,0:4]
Y_np = data[:,-1]

print("Checkpoint #2 - Dataset imported.")


# Plotting all features (X)
def plot_features(X_np):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, figsize=(14,14))

    ax1[0].plot(X_np[:,0])
    ax1[1].plot(X_np[:,1])
    ax2[0].plot(X_np[:,2])
    ax2[1].plot(X_np[:,3])

    ax1[0].grid()
    ax1[1].grid()
    ax2[0].grid()
    ax2[1].grid()

    ax1[0].set_ylabel("a/b")
    ax1[1].set_ylabel("w/b")
    ax2[0].set_ylabel("L_0/b")
    ax2[1].set_ylabel("L_1/b")

    ax1[0].set_title(f"Max: {np.max(X_np[:,0]):>0.3f}, Min: {np.min(X_np[:,0]):>0.3f}, Med: {np.median(X_np[:,0]):>0.3f}")
    ax1[1].set_title(f"Max: {np.max(X_np[:,1]):>0.3f}, Min: {np.min(X_np[:,1]):>0.3f}, Med: {np.median(X_np[:,1]):>0.3f}")
    ax2[0].set_title(f"Max: {np.max(X_np[:,2]):>0.3f}, Min: {np.min(X_np[:,2]):>0.3f}, Med: {np.median(X_np[:,2]):>0.3f}")
    ax2[1].set_title(f"Max: {np.max(X_np[:,3]):>0.3f}, Min: {np.min(X_np[:,3]):>0.3f}, Med: {np.median(X_np[:,3]):>0.3f}")

    fig_iter = 0
    fig_path = f".\\Fig\\X_all_plot-{fig_iter}.png"
    if not os.path.isfile(fig_path):
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")
    else:
        fig_iter += 1
        fig_path = f".\\Fig\\X_all_plot-{fig_iter}.png"
        while os.path.isfile(fig_path):
            fig_iter += 1
            fig_path = f".\\Fig\\X_all_plot-{fig_iter}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")
    print("Saved X_all_plot")

# Plotting all features histogram (X)
def hist_features(X_np):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, figsize=(14,14))

    ax1[0].hist(X_np[:,0], bins=20, density=True)
    ax1[1].hist(X_np[:,1], bins=20, density=True)
    ax2[0].hist(X_np[:,2], bins=20, density=True)
    ax2[1].hist(X_np[:,3], bins=20, density=True)

    ax1[0].grid()
    ax1[1].grid()
    ax2[0].grid()
    ax2[1].grid()

    ax1[0].set_ylabel("a/b")
    ax1[1].set_ylabel("w/b")
    ax2[0].set_ylabel("L_0/b")
    ax2[1].set_ylabel("L_1/b")

    ax1[0].set_title(f"Max: {np.max(X_np[:,0]):>0.3f}, Min: {np.min(X_np[:,0]):>0.3f}, Med: {np.median(X_np[:,0]):>0.3f}")
    ax1[1].set_title(f"Max: {np.max(X_np[:,1]):>0.3f}, Min: {np.min(X_np[:,1]):>0.3f}, Med: {np.median(X_np[:,1]):>0.3f}")
    ax2[0].set_title(f"Max: {np.max(X_np[:,2]):>0.3f}, Min: {np.min(X_np[:,2]):>0.3f}, Med: {np.median(X_np[:,2]):>0.3f}")
    ax2[1].set_title(f"Max: {np.max(X_np[:,3]):>0.3f}, Min: {np.min(X_np[:,3]):>0.3f}, Med: {np.median(X_np[:,3]):>0.3f}")

    fig_iter = 0
    fig_path = f".\\Fig\\X_all_hist-{fig_iter}.png"
    if not os.path.isfile(fig_path):
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")
    else:
        fig_iter += 1
        fig_path = f".\\Fig\\X_all_hist-{fig_iter}.png"
        while os.path.isfile(fig_path):
            fig_iter += 1
            fig_path = f".\\Fig\\X_all_hist-{fig_iter}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")

    print("Saved X_all_hist")

# Plotting target variable (Y)
def plot_target(Y_np):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,7))

    ax[0].plot(Y_np, linewidth=0.5)
    ax[1].hist(Y_np, bins=20, density=True)

    ax[0].grid()
    ax[1].grid()

    ax[0].set_ylabel("y")

    ax[0].set_title(f"Max: {np.max(Y_np):>0.3f}, Min: {np.min(Y_np):>0.3f}, Med: {np.median(Y_np):>0.3f}")
    ax[1].set_title(f"Mean: {np.mean(Y_np):>0.3f}, Std Dev: {np.std(Y_np):>0.3f}, Variance: {np.var(Y_np):>0.3f}")

    fig_iter = 0
    fig_path = f".\\Fig\\Y-{fig_iter}.png"
    if not os.path.isfile(fig_path):
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")
    else:
        fig_iter += 1
        fig_path = f".\\Fig\\Y-{fig_iter}.png"
        while os.path.isfile(fig_path):
            fig_iter += 1
            fig_path = f".\\Fig\\Y-{fig_iter}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")

    print("Saved Y")

# Plotting all features vs Y
def plot_x_y(X_np, Y_np):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, figsize=(14,14))

    ax1[0].plot(X_np[:,0], Y_np)
    ax1[1].plot(X_np[:,1], Y_np)
    ax2[0].plot(X_np[:,2], Y_np)
    ax2[1].plot(X_np[:,3], Y_np)

    ax1[0].grid()
    ax1[1].grid()
    ax2[0].grid()
    ax2[1].grid()

    ax1[0].set_xlabel("a/b")
    ax1[1].set_xlabel("w/b")
    ax2[0].set_xlabel("L_0/b")
    ax2[1].set_xlabel("L_1/b")

    ax1[0].set_ylabel("y")
    ax1[1].set_ylabel("y")
    ax2[0].set_ylabel("y")
    ax2[1].set_ylabel("y")

    fig_iter = 0
    fig_path = f".\\Fig\\X_vs_Y_plot-{fig_iter}.png"
    if not os.path.isfile(fig_path):
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")
    else:
        fig_iter += 1
        fig_path = f".\\Fig\\X_vs_Y_plot-{fig_iter}.png"
        while os.path.isfile(fig_path):
            fig_iter += 1
            fig_path = f".\\Fig\\X_vs_Y_plot-{fig_iter}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")

    print("Saved X_vs_Y_plot")

# Plotting Scatter for all features vs Y
def scatter_x_y(X_np, Y_np):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, figsize=(14,14))

    ax1[0].scatter(X_np[:,0], Y_np)
    ax1[1].scatter(X_np[:,1], Y_np)
    ax2[0].scatter(X_np[:,2], Y_np)
    ax2[1].scatter(X_np[:,3], Y_np)

    ax1[0].grid()
    ax1[1].grid()
    ax2[0].grid()
    ax2[1].grid()

    ax1[0].set_xlabel("a/b")
    ax1[1].set_xlabel("w/b")
    ax2[0].set_xlabel("L_0/b")
    ax2[1].set_xlabel("L_1/b")

    ax1[0].set_ylabel("y")
    ax1[1].set_ylabel("y")
    ax2[0].set_ylabel("y")
    ax2[1].set_ylabel("y")

    fig_iter = 0
    fig_path = f".\\Fig\\X_vs_Y_scatter-{fig_iter}.png"
    if not os.path.isfile(fig_path):
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")
    else:
        fig_iter += 1
        fig_path = f".\\Fig\\X_vs_Y_scatter-{fig_iter}.png"
        while os.path.isfile(fig_path):
            fig_iter += 1
            fig_path = f".\\Fig\\X_vs_Y_scatter-{fig_iter}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")

    print("Saved X_vs_Y_scatter")

# Plotting Contours for all features vs Y
def contour_x_y(X_np, Y_np):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, figsize=(14,14))

    levels= np.linspace(np.min(Y_np), np.max(Y_np), 10)

    ax1[0].tricontourf(X_np[:,0], X_np[:,1], Y_np, levels=levels)
    ax1[1].tricontourf(X_np[:,0], X_np[:,2], Y_np, levels=levels)
    ax2[0].tricontourf(X_np[:,0], X_np[:,3], Y_np, levels=levels)
    ax2[1].tricontourf(X_np[:,1], X_np[:,2], Y_np, levels=levels)

    ax1[0].set_xlabel("a/b")
    ax1[1].set_xlabel("a/b")
    ax2[0].set_xlabel("a/b")
    ax2[1].set_xlabel("w/b")

    ax1[0].set_ylabel("w/b")
    ax1[1].set_ylabel("L_0/b")
    ax2[0].set_ylabel("L_1/b")
    ax2[1].set_ylabel("L_0/b")

    fig_iter = 0
    fig_path = f".\\Fig\\Y_X1_X2_contour-{fig_iter}.png"
    if not os.path.isfile(fig_path):
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")
    else:
        fig_iter += 1
        fig_path = f".\\Fig\\Y_X1_X2_contour-{fig_iter}.png"
        while os.path.isfile(fig_path):
            fig_iter += 1
            fig_path = f".\\Fig\\Y_X1_X2_contour-{fig_iter}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400, format="png")

    print("Saved Y_X1_X2_contour")



# Main code
plot_features(X_np)
plot_x_y(X_np, Y_np)
scatter_x_y(X_np, Y_np)
contour_x_y(X_np, Y_np)

print("Checkpoint #3 - Figures saved.")
