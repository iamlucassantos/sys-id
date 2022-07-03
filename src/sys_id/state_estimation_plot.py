"""Module that plots the output from estate_estimation."""
import pandas as pd
from helpers import F16, REPORT_PATH
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
sns.set_context("talk")


to_save = True  # If figures should be saved
to_plot = True

df = pd.read_csv('state_estimation.csv')
t = df['time']

# Plot with alpha
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(t, F16.z_k[0, :], label=r'$\alpha_{measured}$')
ax.plot(t, df['a_true'], label=r'$\alpha_{true}$')
# ax.plot(t, df['a'], label=r'$\alpha_{estimated}$')
ax.set_xlabel('Time [s]')
ax.set_ylabel('[rad]')
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_PATH/"state_estimation_alpha.pdf")

# Plot with c_m_up
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t, df['C_m_up'])
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$C_{\alpha_{up}}$')
plt.tight_layout()
plt.savefig(REPORT_PATH/"state_estimation_wash.pdf")

fig, ax = plt.subplots(3, 1, figsize=(9, 5))
ax[0].plot(t, df['u'])
ax[0].set_ylabel(r'$u$ [rad]')
ax[1].plot(t, df['v'])
ax[1].set_ylabel(r'$v$ [rad]')
ax[2].plot(t, df['w'])
ax[2].set_ylabel(r'$w$ [rad]')
ax[2].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(REPORT_PATH/"state_estimation_output.pdf")

if to_plot:
    plt.show()
