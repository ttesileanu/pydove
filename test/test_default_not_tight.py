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
# # Test that `bbox_inches` is off by default in the inline backend

# %%
import pydove as dv

# %%
with dv.FigureManager(1, 2) as (fig, axs):
    axs[1].set_visible(False)
    axs[0].patch.set_alpha(0)
    axs[0].text(
        0.0,
        0.5,
        "this figure should have an empty panel to the right -->",
        fontsize="x-large",
    )
    fig.patch.set_facecolor("#537eba")

# %%
with dv.FigureManager() as (_, ax):
    ax.text(
        0.0,
        0.5,
        "the alphabet on the right should be cropped: ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        fontsize="x-large",
    )

# %%
