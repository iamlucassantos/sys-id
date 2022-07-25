"""Module that creates a feed forward neural network"""
import numpy as np
import pandas as pd
import pickle

from helpers import F16, Network
from rich.console import Console


class FF(Network):
    """Feed-forward NN."""

    def __init__(self):
        name = "FF"
        train_func = "tansig"
        self.n_input = None
        self.n_output = None
        self.n_hidden = None
        self.centroids = None
        self.a = 1
        self.b = None
        self.IW = None
        self.LW = None
        self.epoch = None
        self.goal = None
        self.min_grad = None
        self.mu = None
        super().__init__(name, train_func)

    def solve(self, x, IW=None, LW=None):
        """Solve the network."""
        IW = IW if IW is not None else self.IW
        LW = LW if LW is not None else self.LW
        n_measurements = x.shape[0]
        x_bias = x + self.b[0, 0]
        vj = np.zeros((self.n_hidden, n_measurements))

        for idx in range(self.n_input):
            vj += IW[:, [idx]] * x_bias[:, idx]

        yj = 2 / (1 + np.exp(-2 * vj)) - 1

        # Append bias to output of hidden layer
        yj_bias = yj + self.b[1, 0]
        yk = np.sum(LW * yj_bias, axis=0).reshape(-1, 1)

        return yj, yk

    def fit(self, x, y, method="gradient_descent", n_hidden=1, b1=0.5, b2=0.4, epoch=100, goal=0.001, min_grad=0.001,
            mu=1e-6):
        """Fit the network."""
        self.n_input = x.shape[1]
        self.n_output = y.shape[1]
        self.n_hidden = n_hidden
        self.b = np.array([[b1], [b2]])
        self.epoch = epoch
        self.mu = mu
        self.min_grad = min_grad
        # Including bias terms (+1)
        self.IW = np.random.normal(0, 1, (self.n_hidden, self.n_input))
        self.LW = np.random.normal(0, 1, (self.n_hidden, self.n_output))

        output = {}
        if method == "gradient_descent":
            self.gradient_descent(x, y)
        elif method == "levenberg":
            output = self.levenberg(x, y)
        return output

    def jacobian(self, x, y1, e, backpropagation=False):
        """Returns the Jacobian matrix."""
        dE_de = e if backpropagation else 1
        de_dyk = -1
        dyk_dvk = 1
        dvk_dlw = y1

        # Output weight
        dE_dlw = dE_de * de_dyk * dyk_dvk * dvk_dlw.T

        # Input weight
        dvk_dyj = self.LW
        dyj_dvj = 1 - y1 ** 2

        dE_diw_list = []
        for i in range(self.n_input):
            dE_diw_list.append(dE_de * de_dyk * dyk_dvk * dvk_dyj.T * dyj_dvj.T * x[:, [i]])

        dE_diw = np.hstack(dE_diw_list)

        if backpropagation:
            dE_dlw = dE_dlw.sum(axis=0).reshape(-1, 1)
            dE_diw = dE_diw.sum(axis=0).reshape(-1, 2, order="F")
            return dE_diw, dE_dlw,
        else:
            J = np.hstack((dE_dlw, dE_diw))
            return J

    def gradient_descent(self, x, y):
        """Performs gradient descent method in the network."""
        y1, y2 = self.solve(x)
        e, E = self.cost(y, y2)
        cost_list = [E]
        descent = 0

        dE_diw, dE_dlw = self.jacobian(x, y1, e, backpropagation=True)
        for _ in range(self.epoch):
            IW = self.IW - self.mu * dE_diw
            LW = self.LW - self.mu * dE_dlw

            y1_new, y2_new = self.solve(x, IW=IW, LW=LW)
            e_new, E_new = self.cost(y, y2_new)
            gradient = np.abs(np.gradient(cost_list + [E_new])[-1])
            print(gradient)
            if E_new < E and gradient > self.min_grad:
                descent += 1
                self.IW = IW
                self.LW = LW
                y1, y2 = y1_new, y2_new
                e, E = e_new, E_new
                dE_diw, dE_dlw = self.jacobian(x, y1, e, backpropagation=True)
                cost_list.append(E)

            else:
                break
        print(f"Descent: {descent}")

    def levenberg_performance(self, x, y, dw):
        """Returns the performance of the Levenberg-Marquardt method."""

        w = np.vstack((self.IW[:, [0]], self.IW[:, [1]], self.LW))

        w -= dw

        w = w.reshape(-1, 3, order="F")

        IW = w[:, [0, 1]]
        LW = w[:, [2]]

        w_new = {
            'IW': IW,
            'LW': LW,
        }
        # Get estimation with new weights
        y1_new, y2_new = self.solve(x, IW=IW, LW=LW)
        e_new, E_new = self.cost(y, y2_new)

        return y1_new, y2_new, e_new, E_new, w_new

    def update_weights(self, w_new):
        """updates the weights of the network."""
        self.IW = w_new['IW']
        self.LW = w_new['LW']


def main(TO_SAVE=False,
         TRAIN_FF_NEURONS=False,
         TRAIN_LEVENERG_NEURONS=False):
    # Set seed
    np.random.seed(2)

    # Load data
    df = pd.read_csv("state_estimation.csv")
    x1 = df['a_true'].to_numpy()
    x2 = df['b'].to_numpy()
    x = np.array([x1, x2]).T
    y = F16.c_m.reshape(-1, 1)

    # Load validation data
    x_val = np.array([F16.a_val, F16.b_val]).T
    y_val = F16.c_m_val.reshape(-1, 1)

    # Output data
    console = Console()

    all_output = {
        'data': (x, y, x_val, y_val),
        'train_ff_neurons': None
    }

    if TRAIN_FF_NEURONS:
        console.rule("Training FF NN neurons")
        ff = FF()
        ff.fit(x, y, n_hidden=80, epoch=1000)
        y_pred = ff.predict(x)
        print(ff.rms(y, y_pred))
        all_output['train_ff_neurons'] = {
            'y_pred': y_pred
        }

    if TRAIN_LEVENERG_NEURONS:
        console.rule("Training Levenberg-Marquardt NN neurons")
        ff = FF()
        out = ff.fit(x, y, n_hidden=80, epoch=1000, min_grad=1e-17, method='levenberg')
        y_pred = ff.predict(x)
        print(out)
        print(ff.rms(y, y_pred))

    if TO_SAVE:
        pickle.dump(all_output, open("ff_data.pkl", "wb"))


if __name__ == '__main__':
    main(TO_SAVE=False,
         TRAIN_FF_NEURONS=False,
         TRAIN_LEVENERG_NEURONS=True)
