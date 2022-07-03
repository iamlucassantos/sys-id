"""Module that implements an ordinary least squares estimator."""
import pandas as pd
import numpy as np
from helpers import F16

# Load training data
df = pd.read_csv("state_estimation.csv")
x1 = df['a_true'].to_numpy().reshape(-1, 1)
x2 = df['b'].to_numpy().reshape(-1, 1)
y = F16.c_m.reshape(-1, 1)

# Load validation data
x1_val = F16.a_val.reshape(-1, 1)
x2_val = F16.b_val.reshape(-1, 1)
y_val = F16.c_m_val.reshape(-1, 1)


def build_a(vec_1: np.ndarray, vec_2: np.ndarray, pol_ord: int):
    """Builds matrix A for two variables."""
    ones = np.ones(vec_1.shape)
    A = np.zeros(1)
    if pol_ord == 1:
        A = np.block([ones, vec_1, vec_2])
    elif pol_ord == 2:
        A = np.block([ones, vec_1, vec_2, vec_1 ** 2, vec_2 ** 2, vec_1 * vec_2])
    elif pol_ord == 3:
        A = np.block(
            [ones, vec_1, vec_2, vec_1 ** 2, vec_2 ** 2, vec_1 * vec_2, vec_1 ** 3, vec_2 ** 3, vec_1 ** 2 * vec_2,
             vec_1 * vec_2 ** 2])
    elif pol_ord == 4:
        A = np.block(
            [ones, vec_1, vec_2, vec_1 ** 2, vec_2 ** 2, vec_1 * vec_2, vec_1 ** 3, vec_2 ** 3,
             vec_1 ** 2 * vec_2, vec_1 * vec_2 ** 2, vec_1 ** 4, vec_2 ** 4, vec_1 ** 3 * vec_2,
             vec_1 * vec_2 ** 3, vec_1 ** 2 * vec_2 ** 2]
        )

    return A


def solve_ols(A: np.ndarray, y=np.ndarray):
    return np.linalg.inv(A.T @ A) @ A.T @ y


df_train = {}
df_train['y'] = y.flatten()
df_train['x1'] = x1.flatten()
df_train['x2'] = x2.flatten()

df_val = {}
df_val['y'] = y_val.flatten()
df_val['x1'] = x1_val.flatten()
df_val['x2'] = x2_val.flatten()

# Create A matrix
covariance = {
    'train': [],
    'val': []
}
for i in range(1, 5):
    A = build_a(x1, x2, i)
    A_val = build_a(x1_val, x2_val, i)
    theta_ols = solve_ols(A, y)
    y_hat = A @ theta_ols
    y_hat_val = A_val @ theta_ols

    # Get error
    e_train = y - y_hat
    df_train[f"y_{i}"] = y_hat.flatten()
    df_train[f"e_{i}"] = (e_train ** 2).flatten()

    e_val = y_val - y_hat_val
    df_val[f"y_{i}"] = y_hat_val.flatten()
    df_val[f"e_{i}"] = (e_val ** 2).flatten()

    # Get variance
    cov_train = (e_train.T @ e_train) / (len(y) - i) * np.linalg.inv(A.T @ A)
    cov_val = (e_val.T @ e_val) / (len(y) - i) * np.linalg.inv(A_val.T @ A_val)

    covariance['train'].append(cov_train)
    covariance['val'].append(cov_val)
    print(f"Order {i}: {theta_ols}")


df_train = pd.DataFrame.from_dict(df_train)
df_val = pd.DataFrame.from_dict(df_val)

# Save data
df_train.to_csv("data/ols_train.csv")
df_val.to_csv("data/ols_val.csv")
