"""Module that performs the estate estimation."""

import control.matlab as c
import numpy as np
from helpers import F16
from helpers import rk4

# Set seed
np.random.seed(7)

#####################
# 1 Simulation set up
#####################

# Simulation parameters
n = F16.n_states  # State dimension
nm = F16.n_measurements  # Number of measurements
n_samples = F16.n_samples  # Number of samples
dt = 0.01  # Time step [s]
epsilon = 10 ** (-10)  # IEKF threshold TODO
max_iterations = 100  # Number maximum of iterations

# Configurations
to_plot = False  # If plots are desired
to_save = False  # If plots should be saved
to_iterate = True  # If IEFK should be used

######################
# 2 Set initial values
######################

# Define initial values
E_x_0 = np.array([[F16.V_m[0]], [0], [0], [0]])  # Initial estimate of optimal value of x_k1_k1
std_x_0 = np.array([100] * 4)  # Initial standard deviation of state prediction error
P_0 = np.diag(std_x_0) ** 2  # Initial covariance of state prediction error

# System noise statistics
E_w = np.zeros(n)  # Bias of system noise (no bias in noise)
std_w = np.array([1e-3] * 3 + [0])  # Standard deviation of system noise
Q = np.diag(std_w) ** 2  # Variance of system noise
w_k = np.diag(std_w) @ np.random.normal(size=(n, n_samples)) + np.diag(E_w) @ np.ones(
    (n, n_samples)
)  # System noise TODO: Check if should be Q

# Measurement noise statistics
E_v = np.zeros(nm)  # Bias of measurement noise
std_v = np.array([0.035, 0.013, 0.110])  # Standard deviation of measurement noise
R = np.diag(std_v) ** 2  # Variance of measurement noise
v_k = np.diag(std_v) @ np.random.normal(size=(nm, n_samples)) + np.diag(E_v) @ np.ones(
    (nm, n_samples)
)  # Measurement noise

#######################
# 3 Start Kalman filter
#######################

# Initialize Extended Kalman Filter
t_k = 0
t_k1 = dt

# Initialize state estimation and error covariance matrix
x_k1_k1 = E_x_0.copy()  # x(0|0) = E(x_0)
P_k1_k1 = P_0.copy()  # P(0|0) = P(0)

# Run the filter
for k in range(0, n_samples):
    # 1. One step ahead prediction: X_k+1_k
    t, x_k1_k = rk4(F16.calc_f, x_k1_k1.copy(), F16.u_k[:, k], [t_k, t_k1])

    # 2. Calculate Jacobians: Phi_k+1_k, Gamma_k+1_k
    Fx = F16.calc_Fx()

    # 3. Discretize state transition matrix
    ss_B = c.ss(Fx, F16.B, np.zeros((F16.n_measurements, F16.n_states)), 0)
    ss_G = c.ss(Fx, F16.G, np.zeros((F16.n_measurements, F16.n_states)), 0)
    Psi = np.array(c.c2d(ss_B, dt).B)
    Phi = np.array(c.c2d(ss_G, dt).A)
    Gamma = np.array(c.c2d(ss_G, dt).B)

    # 4. Covariance matrix of state prediction error: P_k+1_k
    P_k1_k = Phi @ P_k1_k1 @ Phi.T + Gamma @ Q @ Gamma.T

    if to_iterate:
        eta_2 = x_k1_k.copy()
        error = 2 * epsilon
        n_iterations = 0

        while error > epsilon and n_iterations < max_iterations:
            n_iterations += 1
            eta_1 = eta_2.copy()

            # 5. Recalculate the Jacobian d/dx(h(x))
            Hx = F16.calc_Hx(0, eta_1, F16.u_k[:, k])

            # Prediction of observation (validation)
            z_k1_k = F16.calc_h(0, eta_1, F16.u_k[:, k])

            # Covariance matrix of observation error (validation)
            P_zz = Hx @ P_k1_k @ Hx.T + R

            # Standard deviation of observation error (validation)
            std_z = np.diag(P_zz) ** 2

            # 6. Kalman gain recalculation
            K = P_k1_k @ Hx.T @ np.linalg.inv(P_zz)

            # 7. Measurement update
            eta_2 = x_k1_k + K @ (F16.z_k[:, [k]] - z_k1_k - Hx @ (x_k1_k - eta_1))

            error = np.linalg.norm(eta_2 - eta_1) / np.linalg.norm(eta_1)

        # 8. Covariance matrix of estate estimation error P_k+1_k+1
        P_k1_k1 = (np.identity(F16.n_states) - K @ Hx) @ P_k1_k @ (np.identity(F16.n_states) - K @ Hx).T + \
                  K @ R @ K.T

        # standard deviation of state estimation error (validation)
        std_x_cor = np.diag(P_k1_k1)**2

        # Next step
        t_k = t_k1
        t_k1 += dt