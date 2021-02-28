# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: pydove
#     language: python
#     name: pydove
# ---

# %% [markdown]
# # Test `regplot`

# %%
# %load_ext autoreload
# %autoreload 2

import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

import pydove as dv

# %% [markdown]
# ## Basic usage

# %%
n = 100
alpha = 3.0
beta = -0.15
sigma = 1.0

rng = np.random.default_rng(0)
x = np.linspace(0, 1, n)
y = alpha * x + beta + sigma * rng.normal(size=n)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

sns.regplot(x=x, y=y, ax=ax1)
res = dv.regplot(x=x, y=y, ax=ax2)

sns.despine(offset=10, ax=ax1)
sns.despine(offset=10, ax=ax2)

# %% [markdown]
# ## Custom plotting options

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

kwargs = {"scatter_kws": {"color": "r"}, "line_kws": {"color": "k"}}

sns.regplot(x=x, y=y, **kwargs, ax=ax1)
res = dv.regplot(x=x, y=y, **kwargs, ax=ax2)

sns.despine(offset=10, ax=ax1)
sns.despine(offset=10, ax=ax2)

# %% [markdown]
# ## `pydove`-specific plotting options

# %%
fig, ax = plt.subplots()

res = dv.regplot(x=x, y=y, ci_kws={"fc": "r"}, n_points=2, ax=ax)

sns.despine(offset=10, ax=ax)

# %% [markdown]
# ## Robustness to few data points

# %% [markdown]
# `pydove.regplot` uses `statsmodels` for confidence-interval calculations. `seaborn` uses custom code. The two are similar but not exactly the same, especially when working with very few data points.

# %%
n_max = 5
fig, axs = plt.subplots(
    n_max, 2, figsize=(12, 4 * n_max), sharey=True, tight_layout=True
)

for i, crt_axs in enumerate(axs):
    crt_x = x[:i]
    crt_y = y[:i]

    sns.regplot(x=crt_x, y=crt_y, ax=crt_axs[0])
    dv.regplot(crt_x, crt_y, ax=crt_axs[1])

    for k, ax in enumerate(crt_axs):
        crt_name = ["seaborn", "pydove"][k]
        ax.set_title(f"{i} points, {crt_name}")
        sns.despine(offset=10, ax=ax)

# %% [markdown]
# ## Usage with a `DataFrame`

# %%
df = pd.DataFrame({"x1": x, "x2": y})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

sns.regplot(x="x1", y="x2", data=df, ax=ax1)
res = dv.regplot("x1", "x2", df, ax=ax2)

sns.despine(offset=10, ax=ax1)
sns.despine(offset=10, ax=ax2)

# %% [markdown]
# ## Check turning on or off the display of the scatter plot or fit line, and label

# %%
fig, axs = plt.subplots(4, 2, figsize=(12, 16), tight_layout=True)

for i_scatter in range(2):
    crt_scatter = i_scatter != 0
    for i_fit in range(2):
        crt_fit = i_fit != 0

        i = i_scatter * 2 + i_fit

        sns.regplot(
            x=x, y=y, scatter=crt_scatter, fit_reg=crt_fit, label="fit", ax=axs[i, 0]
        )
        dv.regplot(
            x=x, y=y, scatter=crt_scatter, fit_reg=crt_fit, label="fit", ax=axs[i, 1]
        )
        
        if i > 0:
            axs[i, 0].legend(frameon=False)
            axs[i, 1].legend(frameon=False)

        axs[i, 0].set_title(f"seaborn, scatter={crt_scatter}, fit_reg={crt_fit}")
        axs[i, 1].set_title(f"pydove, scatter={crt_scatter}, fit_reg={crt_fit}")

for crt_axs in axs:
    for ax in crt_axs:
        sns.despine(offset=10, ax=ax)

# %% [markdown]
# ## Check direct color and marker options

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

kwargs = {"color": "r", "marker": "+"}

sns.regplot(x=x, y=y, **kwargs, ax=ax1)
res = dv.regplot(x=x, y=y, **kwargs, ax=ax2)

sns.despine(offset=10, ax=ax1)
sns.despine(offset=10, ax=ax2)

# %% [markdown]
# ## Check CI size

# %%
ci_vals = [None, 40, 90, 99]
n_ci = len(ci_vals)

fig, axs = plt.subplots(n_ci, 2, figsize=(12, n_ci * 4), tight_layout=True)

for i, ci in enumerate(ci_vals):
    crt_axs = axs[i]

    sns.regplot(x=x, y=y, ci=ci, ax=crt_axs[0])
    res = dv.regplot(x=x, y=y, ci=ci, ax=crt_axs[1])

    sns.despine(offset=10, ax=crt_axs[0])
    sns.despine(offset=10, ax=crt_axs[1])

    crt_axs[0].set_title(f"seaborn, CI={ci}")
    crt_axs[1].set_title(f"pydove, CI={ci}")

# %% [markdown]
# ## Check truncate

# %%
fig, axs = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)

for i, crt_axs in enumerate(axs):
    sns.regplot(x=x, y=y, truncate=i == 0, ax=crt_axs[0])
    res = dv.regplot(x=x, y=y, truncate=i == 0, ax=crt_axs[1])

    sns.despine(offset=10, ax=crt_axs[0])
    sns.despine(offset=10, ax=crt_axs[1])

    crt_axs[0].set_title(f"seaborn, truncate={i == 0}")
    crt_axs[1].set_title(f"pydove, truncate={i == 0}")

# %% [markdown]
# ## Check higher order fits

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

order = 3
sns.regplot(x=x, y=y, order=order, ax=ax1)
res = dv.regplot(x=x, y=y, order=order, ax=ax2)

sns.despine(offset=10, ax=ax1)
sns.despine(offset=10, ax=ax2)

# %% [markdown]
# ## Check jitter

# %%
fig, axs = plt.subplots(
    4, 2, figsize=(12, 16), sharex=True, sharey=True, tight_layout=True
)

for i, (ax1, ax2) in enumerate(axs):
    kwargs = {"x_jitter": 0.3, "y_jitter": 1.0}
    if i > 0:
        kwargs["seed"] = [0, np.random.default_rng(1), np.random.RandomState(2)][i - 1]

    crt_name = ["no seed", "int", "default_rng", "RandomState"][i]

    sns.regplot(x=x, y=y, **kwargs, ax=ax1)
    res = dv.regplot(x=x, y=y, **kwargs, ax=ax2)

    ax1.set_title("seaborn, " + crt_name)
    ax2.set_title("pydove, " + crt_name)

    sns.despine(offset=10, ax=ax1)
    sns.despine(offset=10, ax=ax2)

# %% [markdown]
# ## Check logx

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

sns.regplot(x=x + 0.1, y=y, logx=True, ax=ax1)
res = dv.regplot(x=x + 0.1, y=y, logx=True, ax=ax2)

sns.despine(offset=10, ax=ax1)
sns.despine(offset=10, ax=ax2)

# %% [markdown]
# ## Check logx with polynomial regression

# %% [markdown]
# Seaborn doesn't do this.

# %%
fig, ax = plt.subplots()

order = 3
res = dv.regplot(x=x + 0.1, y=y, logx=True, order=order, ax=ax)

sns.despine(offset=10, ax=ax)

# %% [markdown]
# ## Check x_estimator

# %%
x_unique = np.linspace(0, 5, 10)
y_means = -2.5 * x_unique + 0.6

rng = np.random.default_rng(1)
y_per_x = 8
y_values = rng.normal(loc=y_means, size=(y_per_x, len(y_means))).T.ravel()
x_values = np.repeat(x_unique, y_per_x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

sns.regplot(x=x_values, y=y_values, x_estimator=np.mean, ax=ax1)
dv.regplot(x=x_values, y=y_values, x_estimator=np.mean, ax=ax2)

sns.despine(offset=10, ax=ax1)
sns.despine(offset=10, ax=ax2)

# %% [markdown]
# ## Test separate `scatter` function

# %%
df = pd.DataFrame({"x1": x, "x2": y})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)

dv.scatter("x1", "x2", df, ax=ax1)

dv.scatter("x1", "x2", df, s=4, c="gray", ax=ax2)
dv.scatter(x, "x2", df, x_jitter=0.02, y_jitter=0.08, ax=ax2)

ax2.set_title("with jitter")

sns.despine(offset=10, ax=ax1)
sns.despine(offset=10, ax=ax2)

# %%
fig, ax = plt.subplots()

dv.scatter(x=x_values, y=y_values, x_estimator=np.mean, ax=ax)

sns.despine(offset=10, ax=ax)

# %% [markdown]
# ## Test separate `fitplot` function

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4), tight_layout=True)

dv.fitplot(res, ax=ax1)
dv.fitplot(x_range=(-0.5, 1.5), fit_results=res, ax=ax2)
dv.fitplot(x=np.linspace(-0.5, 1.5, 4), fit_results=res, ax=ax3)

for ax in [ax1, ax2, ax3]:
    sns.despine(offset=10, ax=ax)

# %% [markdown]
# ## Test `fitplot` with `polyfit`

# %%
fig, ax = plt.subplots()

ax.scatter(x, y)
poly_res = dv.polyfit(x, y, order=3)
dv.fitplot(poly_res, ax=ax)

sns.despine(offset=10, ax=ax)

# %% [markdown]
# ## Test speed

# %%
rng = np.random.default_rng(0)
big_n = 100_000
big_x = np.linspace(0, 1, big_n)
big_y = alpha * big_x + beta + sigma * rng.normal(size=big_n)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

t0 = time.time()
sns.regplot(x=big_x, y=big_y, scatter_kws={"alpha": 0.02}, ax=ax1)
t1 = time.time()
print(f"sns.regplot took {t1 - t0:.2f} seconds.")
res = dv.regplot(x=big_x, y=big_y, scatter_kws={"alpha": 0.02}, ax=ax2)
t2 = time.time()
print(f"dv.regplot took {t2 - t1:.2f} seconds.")

sns.despine(offset=10, ax=ax1)
sns.despine(offset=10, ax=ax2)

# %%
