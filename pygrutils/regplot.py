""" Define a function that combines a scatter plot with a regression curve. """


import copy

import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Optional, Sequence, Tuple


def regplot(
    x: Sequence = None,
    y: Sequence = None,
    n_points: int = 100,
    scatter_kws: Optional[dict] = None,
    line_kws: Optional[dict] = None,
    ci_kws: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
):
    """ Version of Seaborn `regplot` that returns fit results.
    
    Parameters
    ----------
    x
        Data for x-axis (independent variable).
    y
        Data for y-axis (dependent variable).
    n_points
        Number of points to use for drawing the fit line and confidence interval.
    scatter_kws
        Additional keyword arguments to pass to `plt.scatter` for the scatter plot.
    line_kws
        Additional keyword arguments to pass to `plt.plot` for the fit line.
    ci_kws
        Additional keyword arguments to pass to `plt.fillbetween` for the confidence
        interval.
    ax
        Axes object to draw the plot onto, otherwise uses `plt.gca()`.

    Returns the results from `sm.OLS.fit()`.
    """
    # handle some defaults
    ax = plt.gca() if ax is None else ax

    # calculate best-fit line and interval
    x_fit = sm.add_constant(x)
    fit_results = sm.OLS(y, x_fit).fit()

    eval_x = sm.add_constant(np.linspace(np.min(x), np.max(x), n_points))
    pred = fit_results.get_prediction(eval_x)

    # draw the fit line and error interval
    ci_kws = {} if ci_kws is None else ci_kws
    ci_kws.setdefault("alpha", 0.15)
    ax.fill_between(
        eval_x[:, 1],
        pred.predicted_mean - 2 * pred.se_mean,
        pred.predicted_mean + 2 * pred.se_mean,
        **ci_kws,
    )
    line_kws = {} if line_kws is None else line_kws
    if "lw" not in line_kws and "linewidth" not in line_kws:
        line_kws["lw"] = 2.0
    h = ax.plot(eval_x[:, 1], pred.predicted_mean, **line_kws)

    scatter_kws = {} if scatter_kws is None else scatter_kws
    scatter_kws.setdefault("alpha", 0.8)
    if "c" not in scatter_kws and "color" not in scatter_kws:
        scatter_kws.setdefault("c", h[0].get_color())
    ax.scatter(x, y, **scatter_kws)

    return fit_results
