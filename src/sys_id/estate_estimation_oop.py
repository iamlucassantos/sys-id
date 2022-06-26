import numpy as np
from helpers import training_data
from kalman_filter import Kalman


# Set seed
np.random.seed(7)

# Configurations
to_plot = False  # If plots are desired
to_save = False  # If plots should be saved

# Create KF object
kf = Kalman(
    training_data,
    dt=0.01,
    epsilon=1e-10,  # IEKF threshold TODO
    max_iterations=100,
)

# Set initial values
kf.set_initial(
    E_x_0=np.array([training_data.V_m[0], 0, 0, 0]), std_x_0=np.array([100] * 4)
)

# Set system noise statistics
kf.set_system_noise(E_w=np.zeros(kf.n_states), std_w=np.array([1e-3] * 3 + [0]))

# Set measurement noise statistics
kf.set_measurement_noise(
    E_v=np.zeros(kf.n_measurements), std_v=np.array([0.035, 0.013, 0.110])
)

# Run Kalman Filter
kf.run()
