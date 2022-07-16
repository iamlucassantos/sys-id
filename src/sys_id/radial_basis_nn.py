"""Module that creates the radial basis function neural network."""
from scipy.io import savemat
import numpy as np
import pandas as pd
from helpers import F16
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from scipy.spatial import Delaunay


class Network:

    def __init__(self, name):
        self.name = name

    @staticmethod
    def rms(v1, v2):
        """Calculate the root mean square error between two vectors."""
        return np.sqrt(np.mean((v1 - v2) ** 2))

    def simNet(self, x):
        """Python version of simNet.m"""
        n_input = self.n_input
        n_hidden = self.n_hidden
        n_measurements = x.shape[0]
        IW = self.IW
        if self.name == 'rbf':
            V1 = np.zeros((n_hidden, n_measurements))

            for i in range(n_input):
                V1 += (IW[:, [i]] * x[:, i] - IW[:, [i]] * self.centroids[:, [i]]) ** 2

            Y1 = np.exp(-V1)

            Y2 = self.LW @ Y1

        return Y1, Y2


class RBF(Network):
    """Creates a radial basis function neural network."""

    def __init__(self):
        """Initializes the RBF network."""
        name = 'rbf'
        self.n_input = None
        self.n_output = None
        self.n_hidden = None
        self.centroids = None
        self.IW = None
        self.LW = None
        super().__init__(name)

    def get_centroids(self, x, random_state=1):
        kmeans = KMeans(n_clusters=self.n_hidden, random_state=random_state).fit(x)
        return kmeans.cluster_centers_

    def fit(self, x, y, n_hidden=1):
        """Fits the RBF network."""
        # Get centroids of data
        self.x = x
        self.y = y
        self.n_input = x.shape[1]
        self.n_output = y.shape[1]
        self.n_hidden = n_hidden
        self.IW = np.ones((n_hidden, self.n_input)) / np.std(x, axis=0)
        self.LW = np.random.normal(0, 1, self.n_hidden)
        self.centroids = self.get_centroids(x)

        # Get output of hidden layer
        y1, _ = self.simNet(x)
        y1 = y1.T
        # Usine least square to determine weights of output layer
        w = (np.linalg.inv(y1.T @ y1) @ y1.T) @ y
        self.LW = w.T

    def predict(self, x):
        """Predicts the output of the RBF network."""
        _, y = self.simNet(x)
        y = y.T
        return y

    def plot_centroids(self):
        fig, ax = plt.subplots()
        ax.plot(self.x[:, 0], self.x[:, 1], 'o')
        ax.plot(self.centroids[:, 0], self.centroids[:, 1], 'x')
        plt.show()

    def plot_surface(self, x, y):
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

    def save_mat(self):
        """Saves as matlab stuct."""
        rbf = dict()
        rbf['name'] = np.array([[self.name]], dtype=object)
        rbf['centers'] = self.centroids
        rbf['IW'] = self.IW
        rbf['LW'] = self.LW
        rbf['range'] = np.array([[-1, 1], [-1, 1]])  # TODO
        rbf['trainParam'] = {
            'epoch': 100,
            'goal': 0,
            'min_grad': 1e-10,
            'mu': 1e-3
        }  # TODO
        rbf['x'] = self.x
        rbf['y'] = self.y

        rbf['trainFunct'] = np.array([['radbas'], ['purelin']], dtype=object)  # TODO
        rbf['trainAlg'] = np.array([['trainlm']], dtype=object)  # TODO

        savemat("../assignment_nn/f16_rbf.mat", {'f16_rbf': rbf})


def create_default_rbf(save=False):
    """Creates the rbf structure used by th professor"""
    # rbf = RBF()

    # Create input
    resolution = 0.05
    minXI = -1 * np.ones(2)
    maxXI = 1 * np.ones(2)
    x = np.array([np.arange(minXI[0], maxXI[0] + resolution, resolution)])
    y = np.array([np.arange(minXI[1], maxXI[1] + resolution, resolution)])

    x, y = np.meshgrid(x, y)

    x_eval = np.array([y.T.flatten(), x.T.flatten()])

    rbf = RBF(x_eval.T, np.array([[1]]))

    # Overwrite the centroids
    rbf.centroids = np.array(
        [[-0.900000000000000, 0],
         [-0.700000000000000, 0],
         [-0.500000000000000, 0],
         [-0.300000000000000, 0],
         [-0.100000000000000, 0],
         [0.100000000000000, 0],
         [0.300000000000000, 0],
         [0.500000000000000, 0],
         [0.700000000000000, 0],
         [0.900000000000000, 0]])

    rbf.n_hidden = 10

    # Overwrite the weights
    rbf.IW = np.array([[6.2442] * rbf.n_hidden, [0.6244] * rbf.n_hidden]).T
    rbf.LW = np.array(
        [-0.165029537793434, -0.414146881712120, 0.404277023582498, -0.520573644129355, 0.918965241416011,
         -0.389075595385762, -0.690169083573831, 0.111016838647954, 0.581087378224464, -0.112255412824312])
    return rbf


def main():
    """Main function."""
    # configurations:
    PLOT_OPTIMAL_NEURONS = True

    # Load data
    df = pd.read_csv("state_estimation.csv")
    x1 = df['a_true'].to_numpy()
    x2 = df['b'].to_numpy()
    x = np.array([x1, x2]).T
    y = F16.c_m.reshape(-1, 1)

    # Load validation data
    x_val = np.array([F16.a_val, F16.b_val]).T
    y_val = F16.c_m_val.reshape(-1, 1)

    # Create RBF network

    if PLOT_OPTIMAL_NEURONS:
        n_neurons = np.arange(1, 100)
        error = []
        error_val = []

        # Fit RBF network for different number of neurons
        for i in n_neurons:
            rbf = RBF()
            rbf.fit(x, y, n_hidden=i)

            y_pred = rbf.predict(x)
            y_pred_val = rbf.predict(x_val)

            error.append(rbf.rms(y, y_pred))
            error_val.append(rbf.rms(y_val, y_pred_val))

        # Plot error for different number of neurons
        fig, ax = plt.subplots()
        ax.plot(n_neurons, error)
        ax2 = ax.twinx()
        ax2.plot(n_neurons, error_val, "o--")

        # Get optimal number of neurons
        best_n = np.argmin(error_val) + 1
        print(f"Best number of neurons: {best_n}")

        # Plot error for optimal number of neurons
        rbf.fit(x, y, n_hidden=best_n)
        y_pred = rbf.predict(x)
        y_pred_val = rbf.predict(x_val)

        # Plot training data prediction error
        fig, ax = plt.subplots()
        ax.plot(y_pred[:2000], ".")
        ax.plot(y[:2000], ".")

        # Plot validation data prediction error
        fig, ax = plt.subplots()
        ax.plot(y_pred_val, ".--")
        ax.plot(y_val, ".--")

        # Plot training data and prediction
        rbf.plot_surface(x, y_pred)
        rbf.plot_surface(x, y)

        plt.show()


if __name__ == '__main__':
    main()
