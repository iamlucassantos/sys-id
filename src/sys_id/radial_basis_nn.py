"""Module that creates the radial basis function neural network."""
import pickle

import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.cluster import KMeans

from helpers import F16, Network
from least_square import ols


class RBF(Network):
    """Creates a radial basis function neural network."""

    def __init__(self):
        """Initializes the RBF network."""
        name = 'rbf'
        train_func = 'radbas'
        self.n_input = None
        self.n_output = None
        self.n_hidden = None
        self.centroids = None
        self.a = None
        self.IW = None
        self.LW = None
        self.epoch = None
        self.goal = None
        self.min_grad = None
        self.mu = None
        super().__init__(name, train_func)

    def get_centroids(self, x, random_state=1):
        kmeans = KMeans(n_clusters=self.n_hidden, random_state=random_state).fit(x)
        return kmeans.cluster_centers_

    def jacobian(self, x, y1):
        """Computes the Jacobian matrix."""
        # Output weights error
        de_dy = -1
        dy_dvk = 1
        dvk_dlw = y1
        de_dlw = de_dy * dy_dvk * dvk_dlw

        # Input weights error
        dvk_dyj = self.LW.T
        dyj_dvj = -y1

        de_diw_list = []
        for i in range(self.n_input):
            dvj_diw = 2 * (self.IW[:, [i]] * (x[:, i] - self.centroids[:, [i]]) ** 2)
            de_diw_list.append(de_dy * dy_dvk * dvk_dyj * dyj_dvj * dvj_diw.T)
        de_diw = np.hstack(de_diw_list)

        # Amplitude error
        dyj_da = (y1.T / self.a).T
        de_da = de_dy * dy_dvk * dvk_dyj * dyj_da

        # Centroid error
        de_dc_list = []
        for i in range(self.n_input):
            dvj_dc = 2 * (self.IW[:, [i]] ** 2 * (x[:, i] - self.centroids[:, [i]])) * -1
            de_dc_list.append(de_dy * dy_dvk * dvk_dyj * dyj_dvj * dvj_dc.T)
        de_dc = np.hstack(de_dc_list)

        J = np.hstack((de_diw, de_dlw, de_da, de_dc))

        return J

    def fit(self, x, y, n_hidden=1, method="ols", epoch=100, goal=0, min_grad=1e-10, mu=1e-3):
        """Fits the RBF network."""
        # Get centroids of data
        self.n_input = x.shape[1]
        self.n_output = y.shape[1]
        self.n_hidden = n_hidden
        self.a = np.ones((n_hidden, 1))
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
            A, _ = self.solve(x, a=self.LW)
            self.a = ols(A.T, y)

        elif method == "levenberg":

            # self.IW = np.ones((n_hidden, self.n_input)) / np.std(x, axis=0)
            # A, _ = self.simNet(x, a=self.LW)
            # self.a = ols(A.T, y)

            self.IW = np.ones((n_hidden, self.n_input)) * np.random.normal(0, 1, 2)
            self.LW = np.random.normal(0, 1, (self.n_hidden, 1))
            self.a = np.random.normal(0, 1, (n_hidden, 1))
            output = self.levenberg(x, y)

        return output

    def predict(self, x):
        """Predicts the output of the RBF network."""
        _, y = self.solve(x)
        y = y.T
        return y

    def fit_predict(self, x, y, x_val, y_val, **kwargs):
        """Fits the RBF network and predicts the output of the network."""
        fit_out = self.fit(x, y, **kwargs)
        y_pred = self.predict(x)
        y_pred_val = self.predict(x_val)

        error, cost = self.cost(y, y_pred)
        error_val, cost_val = self.cost(y_val, y_pred_val)

        rms = self.rms(y, y_pred)
        rms_val = self.rms(y_val, y_pred_val)

        output = {
            'fit_out': fit_out,
            'y_pred': y_pred.flatten(),
            'y_pred_val': y_pred_val.flatten(),
            'error': error.flatten(),
            'error_val': error_val.flatten(),
            'cost': cost,
            'cost_val': cost_val,
            'rms': rms,
            'rms_val': rms_val
        }

        return output

    def solve(self, x, IW=None, LW=None, a=None, centroids=None):
        """Python version of simNet.m"""
        n_input = self.n_input
        n_hidden = self.n_hidden
        n_measurements = x.shape[0]
        IW = self.IW if IW is None else IW
        LW = self.LW if LW is None else LW
        a = self.a if a is None else a
        centroids = self.centroids if centroids is None else centroids

        if self.name == 'rbf':
            V1 = np.zeros((n_hidden, n_measurements))

            for i in range(n_input):
                V1 += (IW[:, [i]] * (x[:, i] - centroids[:, [i]])) ** 2

            Y1 = a * np.exp(-V1)

            Y2 = LW.T @ Y1

        return Y1, Y2

    @staticmethod
    def cost(y, y_pred):
        e = y - y_pred.reshape(-1, 1)
        E = np.sum(e ** 2) / 2
        return e, E

    def levenberg(self, x, y):
        """Levenberg-Marquardt algorithm."""
        adaption = 10

        y1, y2 = self.solve(x)
        y1 = y1.T
        e, E = self.cost(y, y2)

        output = {'error': [],
                  'rms': [],
                  'descent': 0,
                  'iterations': 0}

        for epoch in range(self.epoch):
            output['iterations'] += 1
            output["error"].append(E)
            output['rms'].append(self.rms(y, y2))

            J = self.jacobian(x, y1)

            w = np.vstack((self.IW[:, [0]], self.IW[:, [1]], self.LW.reshape(-1, 1), self.a,
                           self.centroids[:, [0]], self.centroids[:, [1]]))

            w -= np.linalg.pinv(J.T @ J + (self.mu * np.identity(J.shape[1]))) @ J.T @ e

            w = w.reshape(-1, 6, order="F")

            IW = w[:, [0, 1]]
            LW = w[:, [2]]
            a = w[:, [3]]
            centroids = w[:, [4, 5]]

            # Get estimation with new weights
            y1_new, y2_new = self.solve(x, IW=IW, LW=LW, a=a, centroids=centroids)
            y1_new = y1_new.T
            e_new, E_new = self.cost(y, y2_new)

            # If error is lower, update weights and increase learning rate
            if E_new < E:
                # print(f"+ {epoch}: Update", self.rms(y, y2), self.rms(y, y2_new))
                output['descent'] += 1
                self.IW = IW
                self.LW = LW
                self.a = a
                self.centroids = centroids
                self.mu *= adaption

                # Update variables
                y1, y2 = y1_new, y2_new
                e, E = e_new, E_new

            else:
                # print(f"- {epoch}: ", self.rms(y, y2), self.rms(y, y2_new), self.mu)
                self.mu /= adaption

            if self.mu < self.min_grad:
                # print(f"Min gradient reached: {self.mu}")
                break

        return output

    def plot_centroids(self):
        fig, ax = plt.subplots()
        ax.plot(self.x[:, 0], self.x[:, 1], 'o')
        ax.plot(self.centroids[:, 0], self.centroids[:, 1], 'x')
        plt.show()


def rbf_ols_optimize_neurons(x, y, x_val, y_val):
    """Optimizes the number of neurons in the hidden layer of the rbf OLS."""
    n_neurons = np.arange(2, 95)

    output = {
        'n_neurons': n_neurons,
        'rms': np.zeros(n_neurons.shape),
        'rms_val': np.zeros(n_neurons.shape),
        'cost': np.zeros(n_neurons.shape),
        'cost_val': np.zeros(n_neurons.shape),
    }

    for idx, n_hidden in enumerate(n_neurons):
        # Fit
        rbf = RBF()
        rbf.fit(x, y, n_hidden=n_hidden)

        # Predict
        y_pred = rbf.predict(x)
        y_pred_val = rbf.predict(x_val)

        # Evaluate
        rms = rbf.rms(y, y_pred)
        rms_val = rbf.rms(y_val, y_pred_val)

        _, cost = rbf.cost(y, y_pred)
        _, cost_val = rbf.cost(y_val, y_pred_val)

        # Save
        output['rms'][idx] = rms
        output['rms_val'][idx] = rms_val
        output['cost'][idx] = cost
        output['cost_val'][idx] = cost_val

    return output


def rbf_lm_optimize_mu(x, y, mu_list):
    """Optimize the rbf LM mu value."""
    rbf = RBF()
    # Get the random state to generate same initial values to all fitting
    random_state = np.random.get_state()
    mu_rms = []
    mu_descent = []
    mu_iterations = []

    for mu in mu_list:
        np.random.set_state(random_state)
        lm_output = rbf.fit(x, y, n_hidden=10, method="levenberg", mu=mu)

        mu_rms.append(lm_output['rms'][-1])
        mu_descent.append(lm_output['descent'])
        mu_iterations.append(lm_output['iterations'])

        print(f"mu: {mu:.2E}, rms: {lm_output['rms'][-1]:.2E}, descent: {lm_output['descent']}")

    output = {
        'mu': mu_list,
        'rms': mu_rms,
        'descent': mu_descent,
        'iterations': mu_iterations,
    }

    return output


def main(TO_SAVE=False,
         RBF_OLS_OPTMIZE_NEURONS=False,
         RBF_OLS_OPTIMAL=False,
         RBF_LM_OPTIMIZE_MU=False,
         RBF_LM_OPTIMAL=False,
         RBF_LM_OPTMIZE_NEURONS=False, ):
    """Main function."""
    # Set seed
    np.random.seed(2)

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

    # Create output dict
    all_output = {
        'data': (x, y, x_val, y_val),
        'ols_optimization': None,
        'ols_optimal': None,
        'lm_optimization_mu': None,
        'lm_optimal': None,
        'lm_optimization_neurons': None,
    }

    if RBF_OLS_OPTMIZE_NEURONS:
        console.rule("[RBF - OLS] Searching for optimal number of neurons with ols")
        output = rbf_ols_optimize_neurons(x, y, x_val, y_val)
        all_output['ols_optimization'] = pd.DataFrame(output)

        # Print optimal number of neurons
        rms = output["rms"]
        rms_val = output["rms_val"]
        arg_lowest_error = np.argmin(rms_val)
        best_n = output['n_neurons'][arg_lowest_error]
        print(f"--> Best number of neurons: {best_n}. "
              f"rms: {rms[arg_lowest_error]:.2E} Validation rms: {rms_val[arg_lowest_error]:.2E}")

    if RBF_OLS_OPTIMAL:
        n_hidden = 85
        console.rule("[RBF - OLS] Fitting RBF network with optimal number of neurons")
        rbf = RBF()
        output = rbf.fit_predict(x, y, x_val, y_val, n_hidden=n_hidden)
        all_output['ols_optimal'] = output
        print(f"--> layers: {n_hidden}, rms: {output['rms']:.2E}, Validation rms: {output['rms_val']:.2E}")

    if RBF_LM_OPTIMIZE_MU:
        console.rule("[RBF - LM] Optimizing Levenberg-Marquardt algorithm mu")

        # Set possible mu values
        mu_list = [10 ** n for n in range(-9, 1)]
        output = rbf_lm_optimize_mu(x, y, mu_list)

        # Print best_my
        best_mu = mu_list[np.argmin(output['rms'])]
        print(f"--> Best mu: {best_mu:.2E}, rms: {np.min(output['rms']):.2E}")

        all_output['lm_optimization_mu'] = pd.DataFrame(output)

    if RBF_LM_OPTIMAL:
        console.rule("[RBF - LM] Fitting RBF network with optimal mu")
        n_hidden = 85
        mu = 1e-4
        # Train large network with best mu
        rbf = RBF()
        output = rbf.fit_predict(x, y, x_val, y_val, n_hidden=n_hidden, method="levenberg", mu=mu)
        all_output['lm_optimal'] = output
        print(f"--> layers: {n_hidden}, mu: {mu}:2E, rms: {output['rms']:.2E}, Validation rms: {output['rms_val']:.2E}")

    if RBF_LM_OPTMIZE_NEURONS:
        console.rule("[RBF - LM] Optimizing number of neurons")
        n_hidden_list = np.arange(10, 201, 10)
        mu = 1e-9
        rms = []
        for n_hidden in n_hidden_list:
            print(n_hidden)
            rbf = RBF()
            output = rbf.fit(x, y, n_hidden=n_hidden, method="levenberg", mu=mu)
            rms.append(output['rms'][-1])

        all_output["lm_optimization_neurons"] = {
            'n_hidden': n_hidden_list,
            'rms': rms,
        }

    if TO_SAVE:
        pickle.dump(all_output, open("rbf_data.pkl", "wb"))


if __name__ == '__main__':
    # SET CONFIGURATIONS
    main(
        TO_SAVE=False,
        RBF_OLS_OPTMIZE_NEURONS=False,
        RBF_OLS_OPTIMAL=True,
        RBF_LM_OPTIMIZE_MU=False,
        RBF_LM_OPTIMAL=True,
        RBF_LM_OPTMIZE_NEURONS=False
    )
