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
# # Test various plotting functions

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import seaborn as sns

import pygrutils as gr

# %%
custom_cmap = gr.gradient_cmap("custom_cmap", "C0", "C1")
with gr.FigureManager(1, 2) as (_, (ax1, ax2)):
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    c = y
    ax1.axhline(0, color="gray", ls=":")
    gr.color_plot(x, y, c, cmap=custom_cmap, ax=ax1)
    ax1.autoscale()

    c = np.linspace(0, 6 * np.pi, 250)
    r = np.linspace(1, 4, len(c)) ** 2
    x = r * np.cos(c)
    y = r * np.sin(c)
    ax2.axhline(0, color="gray", ls=":", lw=0.5)
    ax2.axvline(0, color="gray", ls=":", lw=0.5)
    gr.color_plot(x, y, c, ax=ax2)

    max_r = np.max(r)
    ax2.set_xlim(-max_r, max_r)
    ax2.set_ylim(-max_r, max_r)

    ax2.set_aspect(1)

# %%
