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
#     display_name: pygrutils
#     language: python
#     name: pygrutils
# ---

# %% [markdown]
# # Test colorbar and colormap functions

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import seaborn as sns

import pygrutils as gr

# %% [markdown]
# ## Basic use of `colorbar`

# %%
rng = np.random.default_rng(0)
with gr.FigureManager() as (_, ax):
    n = 500
    x = rng.uniform(size=n)
    y = rng.uniform(size=n)
    h = ax.scatter(x, y, c=y)
    gr.colorbar(h)

# %% [markdown]
# ## `colorbar` options

# %%
rng = np.random.default_rng(0)
with gr.FigureManager() as (_, ax):
    n = 500
    x = rng.uniform(size=n)
    y = rng.uniform(size=n)
    h = ax.scatter(x, y, c=x + y, cmap="Reds")
    gr.colorbar(h, fraction=0.1, pad=1.0, location="left")

# %% [markdown]
# ## Custom gradient color map

# %%
rng = np.random.default_rng(0)
with gr.FigureManager() as (_, ax):
    h = ax.imshow(
        rng.uniform(size=(20, 20)), cmap=gr.gradient_cmap("C0_to_C1", "C0", "C1")
    )
    gr.colorbar(h)

# %%
