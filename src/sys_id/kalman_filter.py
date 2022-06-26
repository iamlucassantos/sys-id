"""Module that defines the KF"""
import numpy as np
from helpers import TrainingData


class Kalman:
    """Class that creates a Kalman Filter."""

    data: TrainingData
    n_states: int
    n_measurements: int
    n_samples: int
    dt: float
    epsilon: float
    max_iterations: int

    def __init__(
        self,
        data: TrainingData,
        n_states: int = 4,
        dt: float = 0.01,
        epsilon: float = 1e-10,
        max_iterations: int = 100,
    ):
        """Creates KF object."""
        self.data = data
        self.n_states = n_states
        self.n_measurements = len(data.z_k)
        self.n_samples = len(data.c_m)
        self.dt = dt
        self.epsilon = epsilon
        self.max_iterations = max_iterations

        # Parameters that are defined later

        # Initial values
        self.E_x_0 = None  # Initial estimate of optimal value of x_k1_k1
        self.std_x_0 = None  # Initial standard deviation of state prediction error
        self.P0 = None  # Initial covariance of state prediction error

        # Initial system noise
        self.E_w = None  # Bias of system noise (no bias in noise)
        self.std_w = None  # Standard deviation of system noise
        self.Q = None  # Variance of system noise
        self.w_k = None  # System noise

        # Initial measurement noise
        self.E_v = None  # Bias of measurement noise
        self.std_v = None  # Standard deviation of measurement noise
        self.R = None  # Variance of measurement noise
        self.v_k = None  # Measurement noise

    def set_initial(self, E_x_0: np.ndarray, std_x_0: np.ndarray):
        """Set the initial values"""
        self.E_x_0 = E_x_0
        self.std_x_0 = std_x_0
        self.P0 = np.diag(std_x_0) ** 2

    def set_system_noise(self, E_w: np.ndarray, std_w: np.ndarray):
        """Initiate system noise vectors"""
        self.E_w = E_w
        self.std_w = std_w
        self.Q = np.diag(std_w) ** 2
        self.w_k = self.create_noise_data(std_w, E_w)

    def set_measurement_noise(self, E_v: np.ndarray, std_v: np.ndarray):
        """Initiate measurement noise vectors"""
        self.E_v = E_v
        self.std_v = std_v
        self.R = np.diag(std_v) ** 2
        self.v_k = self.create_noise_data(std_v, E_v)

    def calc_f(self, t: np.ndarray, x: np.ndarray, u: np.ndarray):
        """Defines the system equation."""
        del t, x
        return u

    def create_noise_data(self, std: np.ndarray, expected_value: np.ndarray):
        """Creates the matrix with noise data."""
        n = len(expected_value)
        N = self.n_samples
        noise = np.diag(std) @ np.random.normal(size=(n, N)) + np.diag(
            expected_value
        ) @ np.ones((n, N))
        return noise

    def run(self):
        """Runs the Kalman filter"""
        # Initialize kalman filter
        t_k = 0
        t_k1 = self.dt

        # Initialize state estimation and error covariance matrix
        x_k1_k1 = self.E_x_0
        P_k1_k1 = self.P0

        for k in range(0, self.n_samples):

            # First prediction: X_k+1_k
            t, x_k1_k = rk4(self.calc_f, x_k1_k1, self.data.u_k[:, k], [t_k, t_k1])

            # Calculate Jacobians: Phi_k+1_k, Gamma_k+1_k


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

    # def create_ss(self):
    #     """Creates KF state space"""
    #     # x_k = [u, v, w, c_a_up]
    #     # z_k = [a_m, b_m, v_m]
    #     # u_k = [u', v', w']
    #
    #     H = np.array([
    #         []
    #     ])
