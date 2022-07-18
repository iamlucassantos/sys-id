"""Module that creates the radial basis function neural network."""
import numpy as np
import pandas as pd

from scipy.io import savemat
from sklearn.cluster import KMeans
from rich.console import Console
from rich.table import Table
import pickle

from helpers import F16
from least_square import ols


# Set seed
np.random.seed(1)

class Network:

    def __init__(self, name):
        self.name = name

    @staticmethod
    def rms(v1, v2):
        """Calculate the root mean square error between two vectors."""
        return np.sqrt(np.mean((v1 - v2) ** 2))

    def simNet(self, x, IW = None, LW = None):
        """Python version of simNet.m"""
        n_input = self.n_input
        n_hidden = self.n_hidden
        n_measurements = x.shape[0]
        IW = self.IW if IW is None else IW
        LW = self.LW if LW is None else LW
        if self.name == 'rbf':
            V1 = np.zeros((n_hidden, n_measurements))

            for i in range(n_input):
                V1 += (IW[:, [i]] * (x[:, i] - self.centroids[:, [i]])) ** 2

            Y1 = np.exp(-V1)

            Y2 = LW.T @ Y1

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
        self.epoch = None
        self.goal = None
        self.min_grad = None
        self.mu = None
        super().__init__(name)

    def get_centroids(self, x, random_state=1):
        kmeans = KMeans(n_clusters=self.n_hidden, random_state=random_state).fit(x)
        return kmeans.cluster_centers_

    def fit(self, x, y, n_hidden=1, method="ols",
            epoch=100, goal=0, min_grad=1e-10, mu=1e-8):
        """Fits the RBF network."""
        # Get centroids of data
        self.n_input = x.shape[1]
        self.n_output = y.shape[1]
        self.n_hidden = n_hidden
        self.IW = np.ones((n_hidden, self.n_input))
        self.LW = np.ones((self.n_hidden, self.n_output))
        self.centroids = self.get_centroids(x)
        self.epoch = epoch
        self.goal = goal
        self.min_grad = min_grad
        self.mu = mu

        # Get output of hidden layer

        # Usine least square to determine weights of output layer
        output = {}
        if method == "ols":
            self.IW = np.ones((n_hidden, self.n_input)) / np.std(x, axis=0)
            A, _ = self.simNet(x)
            self.LW = ols(A.T, y)


        elif method == "levenberg":
            self.IW = np.random.normal(0, 1, (n_hidden, self.n_input))
            self.LW = np.random.normal(0, 1, (self.n_hidden, 1))
            output = self.levenberg(x, y)

        return output
    def predict(self, x):
        """Predicts the output of the RBF network."""
        _, y = self.simNet(x)
        y = y.T
        return y

    @staticmethod
    def cost(y, y_pred):
        e = y - y_pred.reshape(-1, 1)
        E = np.sum(e ** 2) / 2
        return e, E

    def levenberg(self, x, y):
        """Levenberg-Marquardt algorithm."""
        adaption = 2

        output = {'error': []}

        for epoch in range(self.epoch):
            y1, y2 = self.simNet(x)
            y1 = y1.T
            e, E = self.cost(y, y2)
            output["error"].append(E)

            # Output weights
            de_dy = e * -1
            dy_dvk = 1
            dvk_dlw = y1
            de_dlw = de_dy * dy_dvk * dvk_dlw

            # Input weights
            dvk_dyj = self.LW.T
            dyj_dvj = -y1

            de_diw_list = []
            for i in range(self.n_input):
                dvj_diw = 2*(self.IW[:, [i]] * (x[:, i] - self.centroids[:, [i]])**2)
                de_diw_list.append(de_dy * dy_dvk * dvk_dyj * dyj_dvj * dvj_diw.T)

            de_diw = np.hstack(de_diw_list)

            J = np.hstack((de_diw, de_dlw))


            w = np.vstack((self.IW[:, [0]], self.IW[:, [1]], self.LW.reshape(-1, 1)))
            # print(f"Epoch {epoch}: {w}")

            w -= np.linalg.pinv(J.T@J + (self.mu * np.identity(J.shape[1])))@J.T@e



            w = w.reshape(-1, 3, order="F")

            IW = w[:, [0, 1]]
            LW = w[:, [2]]

            # Get estimation with new weights
            _, y2_new = self.simNet(x, IW=IW, LW=LW)
            _, E_new = self.cost(y, y2_new)
            # If error is lower, update weights and increase learning rate
            if E_new < E:
                print(f"+ {epoch}: Update", E)
                self.mu *= adaption
                self.IW = IW
                self.LW = LW
            else:
                print(f"- {epoch}: ", E, self.mu)

                self.mu *= adaption

        return output

    def plot_centroids(self):
        fig, ax = plt.subplots()
        ax.plot(self.x[:, 0], self.x[:, 1], 'o')
        ax.plot(self.centroids[:, 0], self.centroids[:, 1], 'x')
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
            'epoch': 1000,
            'goal': 0,
            'min_grad': 1e-10,
            'mu': 1e-1
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
    # configurations:
    TO_SAVE = True
    RBF_OPTIMAL_NEURONS = False
    RBF_LEVENBERG = True

    # Create output dict
    output = {
        'rbf': dict(),
        'levenberg': dict(),
    }

    console = Console()
    # Load data
    df = pd.read_csv("state_estimation.csv")
    x1 = df['a_true'].to_numpy()
    x2 = df['b'].to_numpy()
    x = np.array([x1, x2]).T
    y = F16.c_m.reshape(-1, 1)

    # Load validation data
    x_val = np.array([F16.a_val, F16.b_val]).T
    y_val = F16.c_m_val.reshape(-1, 1)

    output["y"] = y
    output["y_val"] = y_val
    output["x"] = x
    output["x_val"] = x_val

    # Create RBF network
    if RBF_OPTIMAL_NEURONS:
        console.rule("[RBF] Searching for optimal number of neurons with ols")
        n_neurons = np.arange(2, 95)
        output['rbf']['n_neurons'] = n_neurons
        output["rbf"]["error"] = []
        output["rbf"]["error_val"] = []

        # Fit RBF network for different number of neurons
        table = Table(title="RBF NN number of neurons")
        table.add_column("Neurons")
        table.add_column("Train RMS")
        table.add_column("Validation RMS")
        for i in n_neurons:
            rbf = RBF()
            rbf.fit(x, y, n_hidden=i)

            y_pred = rbf.predict(x)
            y_pred_val = rbf.predict(x_val)

            pred_error = rbf.rms(y, y_pred)
            val_error = rbf.rms(y_val, y_pred_val)
            output["rbf"]["error"].append(pred_error)
            output["rbf"]["error_val"].append(val_error)
            table.add_row(str(i), f"{pred_error:.2E}", f"{val_error:.2E}")

        console.print(table)
        # Get optimal number of neurons
        error = output["rbf"]["error"]
        error_val = output["rbf"]["error_val"]
        arg_lowest_error = np.argmin(error_val)
        best_n = n_neurons[arg_lowest_error]
        print(f"--> Best number of neurons: {best_n}. "
              f"Train error: {error[arg_lowest_error]:.2E} Validation error: {arg_lowest_error:.2E}")

        # Results for optimal number of neurons
        rbf.fit(x, y, n_hidden=80)
        y_pred = rbf.predict(x)
        y_pred_val = rbf.predict(x_val)
        output['rbf']['y_pred'] = y_pred
        output['rbf']['y_pred_val'] = y_pred_val

    if RBF_LEVENBERG:
        console.rule("[RBF] Using Levenberg-Marquardt algorithm")
        rbf = RBF()
        out = rbf.fit(x, y, n_hidden=85, method="levenberg")
        y_pred = rbf.predict(x)
        output["levenberg"].update(out)
        output['levenberg']['y_pred'] = y_pred
        # plt.plot(out["error"])
        # plt.show()
        # print(1)

    if TO_SAVE:
        pickle.dump(output, open("rbf_data.pkl", "wb"))


if __name__ == '__main__':
    main()
