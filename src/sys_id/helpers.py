"""Helpers module."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Define paths to data files
ROOT_PATH = Path(__file__).parent.parent
FILES_PATH = ROOT_PATH / "assignment_nn"

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
        self.G = np.zeros((self.n_states, self.n_states))  # Noise input matrix

    @staticmethod
    def calc_f(t: np.ndarray, x: np.ndarray, u: np.ndarray):
        """Defines the state equation."""
        del t, x
        return u

    @staticmethod
    def calc_Fx():
        """Defines Jacobian of the state equation"""
        return np.zeros((4, 4))

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
