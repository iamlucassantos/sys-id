import pickle
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from scipy.spatial import Delaunay
import seaborn as sns

# Plotting settings
sns.set_theme(style="darkgrid")
sns.set_context("talk")

# Settings
TO_SAVE = False
PLOT_SURFACE = False

# Load data
nn_data = pickle.load(open("rbf_data.pkl", "rb"))


def plot_surface(x, y):
    """Plots the surface of the RBF network."""

    points2D = x
    tri = Delaunay(points2D)
    simplices = tri.simplices

    fig = ff.create_trisurf(x=x[:, 0], y=x[:, 1], z=y.flatten(),
                            colormap=['rgb(50, 0, 75)', 'rgb(200, 0, 200)', '#c8dcc8'],
                            show_colorbar=True,
                            simplices=simplices,
                            title="Boy's Surface")
    fig.show()


if nn_data['rbf']:
    print("Plotting rbf results.")
    data = nn_data['rbf']

    # Plot error for different number of neurons
    fig, ax = plt.subplots()
    ax.plot(data["n_neurons"], data["error"], ".--")
    ax.plot(data["n_neurons"], data["error_val"], ".--")
    # ax.set_yscale("log")

    # Plot training data prediction error
    fig, ax = plt.subplots()
    ax.plot(data["y_pred"][:2000], ".")
    ax.plot(nn_data["y"][:2000], ".")

    # Plot validation data prediction error
    fig, ax = plt.subplots()
    ax.plot(data["y_pred_val"], ".--")
    ax.plot(nn_data["y_val"], ".--")

    # Plot training data and prediction
    if PLOT_SURFACE:
        plot_surface(nn_data["x"], data["y_pred"])
        plot_surface(nn_data["x"], nn_data["y"])

if nn_data['levenberg']:
    print("Plotting levenberg results.")
    data = nn_data['levenberg']

    fig, ax = plt.subplots()
    ax.plot(data["error"], ".--")

    # fig, ax = plt.subplots()
    # ax.plot(data["y_pred"], ".--")

    plot_surface(nn_data["x"], data["y_pred"])
    print(data)

plt.show()
