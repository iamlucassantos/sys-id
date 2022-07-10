"""Module that creates the radial basis function neural network."""
from scipy.io import savemat
import numpy as np
import pandas as pd
from helpers import F16, simNet
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class RBF:
    """Creates a radial basis function neural network."""

    def __init__(self, inputs, outputs, n_clusters=2):
        """Initializes the RBF network."""
        self.name = 'rbf'
        self.n_input = inputs.shape[1]
        self.n_measurements = inputs.shape[0]
        self.n_output = outputs.shape[1]
        self.x = inputs
        self.y = outputs
        self.centroids = self.get_centroids(n_clusters)
        self.n_hidden = self.centroids.shape[0]
        self.IW = np.random.normal(0, 1, (self.n_hidden, self.n_input))
        self.LW = np.random.normal(0, 1, self.n_hidden)

    def get_centroids(self, n_clusters, random_state=0):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(self.x)
        return kmeans.cluster_centers_

    def plot_centroids(self):
        fig, ax = plt.subplots()
        ax.plot(self.x[:, 0], self.x[:, 1], 'o')
        ax.plot(self.centroids[:, 0], self.centroids[:, 1], 'x')
        plt.show()

    def plot_surface(self, y):
        """Plots the surface of the RBF network."""
        x1, x2 = np.meshgrid(self.x[:, 0], self.x[:, 1])
        y = y.reshape(1, -1)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(x1, x2, y, cmap='viridis')
        # ax.plot(self.x[:, 0], self.x[:, 1], 'o')
        # ax.plot(self.centroids[:, 0], self.centroids[:, 1], 'x')
        # ax.plot(self.x[:, 0], y, '-')
        plt.show()

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
    df = pd.read_csv("state_estimation.csv")
    x1 = df['a_true'].to_numpy()
    x2 = df['b'].to_numpy()
    x = np.array([x1, x2]).T
    y = F16.c_m.reshape(-1, 1)

    # Create RBF network
    rbf = RBF(x, y, n_clusters=10)
    rbf.save_mat()
    # rbf.plot_centroids()

    y = simNet(rbf)
    create_default_rbf()
    # rbf.plot_surface(y)

    default_rbf = create_default_rbf()
    # default_rbf.save_mat()


if __name__ == '__main__':
    main()

#
