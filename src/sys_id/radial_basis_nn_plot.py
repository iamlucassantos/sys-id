import pickle
import matplotlib.pyplot as plt
import pandas as pd
import plotly.figure_factory as ff
from scipy.spatial import Delaunay
import seaborn as sns

# Plotting settings
sns.set_theme(style="darkgrid")
sns.set_context("talk")

# Settings
TO_SAVE = False
PLOT_SURFACE = True

# Load data
rbf_data = pickle.load(open("rbf_data.pkl", "rb"))
x, y, x_val, y_val = rbf_data["data"]


def plot_surface(x, y, title=""):
    """Plots the surface of the RBF network."""
    if PLOT_SURFACE:
        points2D = x
        tri = Delaunay(points2D)
        simplices = tri.simplices

        fig = ff.create_trisurf(x=x[:, 0], y=x[:, 1], z=y.flatten(),
                                colormap=['rgb(50, 0, 75)', 'rgb(200, 0, 200)', '#c8dcc8'],
                                show_colorbar=True,
                                simplices=simplices,
                                title=title)
        fig.show()


if rbf_data['ols_optimization'] is not None:
    print("[RBF - OLS] Plotting rbf neurons optimization results.")
    data = rbf_data['ols_optimization']

    # Plot error for different number of neurons
    fig, ax = plt.subplots()
    ax.plot(data["n_neurons"], data["rms"], ".--")
    ax.plot(data["n_neurons"], data["rms_val"], ".--")
    # ax.set_yscale("log")

if rbf_data['ols_optimal'] is not None:
    print("[RBF - OLS] Plotting rbf optimal results.")
    data = rbf_data['ols_optimal']

    # Plot training data prediction error
    fig, ax = plt.subplots()
    ax.plot(data["y_pred"][:2000], ".")
    ax.plot(y[:2000], ".")

    # Plot validation data prediction error
    fig, ax = plt.subplots()
    ax.plot(data["y_pred_val"], ".--")
    ax.plot(y_val, ".--")

    # Plot training data and prediction
    plot_surface(x, data["y_pred"], "OLS - Prediction")
    plot_surface(x, y, "Training data")

if rbf_data['lm_optimization_mu'] is not None:
    print("[RBF - LM] Plotting levenberg mu optimization results.")
    data = rbf_data['lm_optimization_mu']

    fig, ax = plt.subplots()
    ax.plot(data["rms"], ".--")

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    sns.set_color_codes("pastel")
    sns.barplot(x='mu', y='iterations', label='iterations', data=data, color='b', ax=ax)
    sns.set_color_codes("muted")
    sns.barplot(x='mu', y='descent', data=data, label='descent', color='b', ax=ax)
    sns.pointplot(x='mu', y='rms', data=data, label='rms', color='r', ax=ax2)

    fig, ax = plt.subplots()
    ax.plot(data['rms'], ".--")

if rbf_data['lm_optimal'] is not None:
    print("[RBF - LM] Plotting levenberg optimal.")
    data = rbf_data['lm_optimal']

    plot_surface(x, data["y_pred"], "LM - Prediction")
    plot_surface(x, y, "Training data")

plt.show()
