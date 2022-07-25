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
ff_data = pickle.load(open("ff_data.pkl", "rb"))
x, y, x_val, y_val = ff_data["data"]

if ff_data['train_ff_neurons'] is not None:
    data = ff_data['train_ff_neurons']

    plot_surface(x, data["y_pred"], "FF - Prediction", PLOT_SURFACE=PLOT_SURFACE)