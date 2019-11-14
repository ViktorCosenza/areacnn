# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from os import path
# -

# ## Count number of pixels task

# +
ROOT_DIR = './polygon_data/results'
AVG_FILE = 'avg_pool/history_avg_pool.csv'
MAX_FILE = 'max_pool/history_max_pool.csv'
SUM_FILE = 'sum_pool/history_sum_pool.csv'
MLP_FILE = 'mlp/history_mlp.csv'

df_avg = pd.read_csv(path.join(ROOT_DIR, AVG_FILE))
df_max = pd.read_csv(path.join(ROOT_DIR, MAX_FILE))
df_sum = pd.read_csv(path.join(ROOT_DIR, SUM_FILE))
df_mlp = pd.read_csv(path.join(ROOT_DIR, MLP_FILE))

df_mlp.head()

# +
metric = "mean_absolute_error"

plt.plot(df_avg[[metric]], '-og', label="avg")
plt.plot(df_max[[metric]], '-or', label="max")
plt.plot(df_sum[[metric]], '-oy', label="sum")
plt.plot(df_mlp[[metric]], '-oc', label="mlp")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel(metric)
plt.title("Task: Count number of pixels")
plt.show()
# -

# ## Calculate area percentage (Covered/Total)

# +
ROOT_DIR = './polygon_data_counts/results'

df_avg = pd.read_csv(path.join(ROOT_DIR, AVG_FILE))
df_max = pd.read_csv(path.join(ROOT_DIR, MAX_FILE))
df_sum = pd.read_csv(path.join(ROOT_DIR, SUM_FILE))
df_avg.head()

# +
plt.plot(df_avg[[metric]], '-og', label="avg")
plt.plot(df_max[[metric]], '-or', label="max")
plt.plot(df_sum[[metric]], '-oy', label="sum")
plt.plot(df_mlp[[metric]], '-oc', label="sum")


plt.legend()
plt.xlabel("Epoch")
plt.ylabel(metric)
plt.title("Task: Estimate area percentage")
plt.show()
# -




