import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from helpers import plot_surface

# Plotting settings
sns.set_theme(style="darkgrid")
sns.set_context("talk")

# Settings
TO_SAVE = False
PLOT_SURFACE = True

# Load data
rbf_data = pickle.load(open("rbf_data.pkl", "rb"))
x, y, x_val, y_val = rbf_data["data"]

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
    plot_surface(x, data["y_pred"], "OLS - Prediction", PLOT_SURFACE=PLOT_SURFACE)
    plot_surface(x, y, "Training data", PLOT_SURFACE=PLOT_SURFACE)

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

    plot_surface(x, data["y_pred"], "LM - Prediction", PLOT_SURFACE=PLOT_SURFACE)
    plot_surface(x, y, "Training data", PLOT_SURFACE=PLOT_SURFACE)

plt.show()
