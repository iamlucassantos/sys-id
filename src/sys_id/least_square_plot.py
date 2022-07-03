import pandas as pd
from helpers import REPORT_PATH
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="darkgrid")
sns.set_context("talk")

df_train = pd.read_csv("data/ols_train.csv")
df_val = pd.read_csv("data/ols_val.csv")

# Plot working space
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(df_train['x1'], df_train['x2'], 'o', label='Training data')
ax.plot(df_val['x1'], df_val['x2'], 'o', label='Validation data')
ax.set_xlabel(r'$\alpha$ [rad]')
ax.set_ylabel(r'$\beta$ [rad]')
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_PATH/"parameter_estimation_domain.pdf")

# Plot pol order
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(np.sqrt(df_train[['e_1', 'e_2', 'e_3', 'e_4']].mean()), '-o', label="Training data")
ax.plot(np.sqrt(df_val[['e_1', 'e_2', 'e_3', 'e_4']].mean()), '-o', label="Validation data")
ax.set_xticklabels([i for i in range(1, 5)])
ax.set_xlabel("Order [-]")
ax.set_ylabel("RMS [-]")
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_PATH/"parameter_estimation_mse.pdf")


fig, ax = plt.subplots(figsize=(9, 5))
ax.boxplot(df_train[['e_1', 'e_2', 'e_3', 'e_4']])


plt.show()