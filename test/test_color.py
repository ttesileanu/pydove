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
# # Test colorbar and colormap functions

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import seaborn as sns

import pydove as dv

# %% [markdown]
# ## Basic use of `colorbar`

# %%
rng = np.random.default_rng(0)
with dv.FigureManager() as (_, ax):
    n = 500
    x = rng.uniform(size=n)
    y = rng.uniform(size=n)
    h = ax.scatter(x, y, c=y)
    dv.colorbar(h)

# %% [markdown]
# ## `colorbar` options

# %%
rng = np.random.default_rng(0)
with dv.FigureManager() as (_, ax):
    n = 500
    x = rng.uniform(size=n)
    y = rng.uniform(size=n)
    h = ax.scatter(x, y, c=x + y, cmap="Reds")
    dv.colorbar(h, fraction=0.1, pad=1.0, location="left")

# %% [markdown]
# ## Custom gradient color map

# %%
rng = np.random.default_rng(0)
with dv.FigureManager() as (_, ax):
    h = ax.imshow(
        rng.uniform(size=(20, 20)), cmap=dv.gradient_cmap("C0_to_C1", "C0", "C1")
    )
    dv.colorbar(h)

# %%
