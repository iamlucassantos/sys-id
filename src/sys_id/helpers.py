"""Helpers module."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from math import atan

# Define paths to data files
ROOT_PATH = Path(__file__).parent.parent
FILES_PATH = ROOT_PATH / "assignment_nn"
REPORT_PATH = ROOT_PATH.parent / "report/figures"

# Define name of data files
TRAINING_FILE = "F16traindata_CMabV_2022.csv"


class Model:
    """Class with Kalman filter helpers"""

    def __init__(self):
        """Creates Kalman object"""
        self.n_states = 4  # Number of states
        self.n_measurements = 3  # Number of measurements
        self.c_m, self.z_k, self.u_k = self.load_training_data()
        self.n_samples = len(self.c_m)
        self.B = np.eye(self.n_states, self.n_measurements)  # Input matrix
        self.B[-1, :] = 0
        self.G = np.zeros((self.n_states, self.n_states))  # Noise input matrix

    @staticmethod
    def calc_f(t: np.ndarray, x: np.ndarray, u: np.ndarray):
        """Defines the state equation."""
        del t, x
        return np.array([u]).T

    @staticmethod
    def calc_Fx():
        """Defines Jacobian of the state equation"""
        return np.zeros((4, 4))

    def calc_Hx(self, t: np.ndarray, x: np.ndarray, u: np.ndarray):
        """Defines the Jacobian H=d/dx(h(x))."""
        del t, u
        u, v, w, C_a_up = x.flatten()

        Hx = np.zeros((self.n_measurements, self.n_states))

        # d/du
        Hx[0, 0] = -w / (u ** 2 + w ** 2) * (1 + C_a_up)
        Hx[1, 0] = -u * v / ((u ** 2 + v ** 2 + w ** 2) * (u ** 2 + w ** 2) ** .5)
        Hx[2, 0] = u / (u ** 2 + v ** 2 + w ** 2) ** .5

        # d/dv
        Hx[0, 1] = 0
        Hx[1, 1] = (u ** 2 + w ** 2) ** .5 / (u ** 2 + v ** 2 + w ** 2)
        Hx[2, 1] = v / (u ** 2 + v ** 2 + w ** 2) ** .5

        # d/dw
        Hx[0, 2] = u / (u ** 2 + w ** 2) * (1 + C_a_up)
        Hx[1, 2] = -u * v / ((u ** 2 + v ** 2 + w ** 2) * (u ** 2 + w ** 2) ** .5)
        Hx[2, 2] = w / (u ** 2 + v ** 2 + w ** 2) ** .5

        # d/dC_a_up
        Hx[0, 3] = atan(w / u)
        Hx[1, 3] = 0
        Hx[2, 3] = 0
        return Hx

    def calc_h(self, t: np.ndarray, x: np.ndarray, u: np.ndarray):
        """Defines the observation matrix."""
        del t, u
        u, v, w, C_a_up = x.flatten()
        h = np.zeros((self.n_measurements, 1))

        h[0, 0] = atan(w / u) * (1 + C_a_up)
        h[1, 0] = atan(v / (u ** 2 + w ** 2) ** .5)
        h[2, 0] = (u ** 2 + v ** 2 + w ** 2) ** .5

        return h

    @staticmethod
    def load_training_data():
        """Loads training data"""
        with open(FILES_PATH / TRAINING_FILE) as csvfile:
            # Define name of fields
            field_names = ["cm", "a_m", "b_m", "V_m", "u_dot", "v_dot", "w_dot"]
            dict_data = {field: [] for field in field_names}

            # Create csv reader
            reader = csv.DictReader(csvfile, fieldnames=field_names)

            # Read rows and append to dict
            for row in reader:
                for field in field_names:
                    dict_data[field].append(row[field])

        c_m = np.array(dict_data["cm"], dtype=float)
        z_k = np.array(
            [dict_data["a_m"], dict_data["b_m"], dict_data["V_m"]], dtype=float
        )
        u_k = np.array(
            [
                dict_data["u_dot"],
                dict_data["v_dot"],
                dict_data["w_dot"],
                [0] * len(dict_data["u_dot"]),
            ],
            dtype=float,
        )

        return c_m, z_k, u_k

    @property
    def V_m(self):
        """Returns V_m"""
        return self.z_k[2, :]


def rk4(func, x_0, u_0, t):
    """Runge kutta integration."""
    t_0 = t[0]
    t_end = t[1]
    w = x_0
    N = 2
    h = (t_end - t_0) / N
    t = t_0
    for _ in range(1, N + 1):
        K1 = h * func(t, w, u_0)
        K2 = h * func(t + h / 2, w + K1 / 2, u_0)
        K3 = h * func(t + h / 2, w + K2 / 2, u_0)
        K4 = h * func(t + h, w + K3, u_0)

        w += (K1 + 2 * K2 + 2 * K3 + K4) / 6
        t += h

    return t, w


# Creates F16 model
F16 = Model()
