""" Define a function that combines a scatter plot with a regression curve. """


import copy

import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Optional, Sequence, Tuple


def regplot(
    x: Union[None, str, pd.Series, Sequence] = None,
    y: Union[None, str, pd.Series, Sequence] = None,
    data: Optional[pd.DataFrame] = None,
    n_points: int = 100,
    dropna: bool = True,
    scatter_kws: Optional[dict] = None,
    line_kws: Optional[dict] = None,
    ci_kws: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
):
    """ Version of Seaborn `regplot` that returns fit results.
    
    Parameters
    ----------
    x
        Data for x-axis (independent variable). This can be the data itself, or a string
        indicating a column in `data`.
    y
        Data for y-axis (dependent variable). This can be the data itself, or a string
        indicating a column in `data`.
    data
        Data in Pandas format. `x` and `y` should be strings indicating which columns to
        use for independent and dependent variable, respectively.
    n_points
        Number of points to use for drawing the fit line and confidence interval.
    dropna
        Drop any observations in which either `x` or `y` is not-a-number.
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
    # convert data to a standard form
    x, y = _standardize_data(x=x, y=y, data=data, dropna=dropna)

    # perform some basic checks
    if len(x) != len(y):
        raise ValueError("Different length x and y..")
    if len(x) == 0:
        # nothing to do
        return
    
    # handle some defaults
    ax = plt.gca() if ax is None else ax

    # calculate best-fit line and interval
    x_fit = sm.add_constant(x)
    fit_results = sm.OLS(y, x_fit).fit()

    if len(x) > 2:
        eval_x = sm.add_constant(np.linspace(np.min(x), np.max(x), n_points))
        pred = fit_results.get_prediction(eval_x)

        # set up keywords for fit line and error interval
        ci_kws = {} if ci_kws is None else ci_kws
        line_kws = {} if line_kws is None else line_kws

        ci_kws.setdefault("alpha", 0.15)
        if "ec" not in ci_kws and "edgecolor" not in ci_kws:
            ci_kws["ec"] = "none"

        # if color is provided in line_kws, use same one in ci_kws (unless overridden)
        color_key = None
        if not any(_ in ci_kws for _ in ["c", "color", "facecolor", "facecolors", "fc"]):
            if "c" in line_kws:
                color_key = "c"
            elif "color" in line_kws:
                color_key = "color"
                
            if color_key is not None:
                ci_kws["facecolor"] = line_kws[color_key]
                
        # draw the fit line and error interval
        ax.fill_between(
            eval_x[:, 1],
            pred.predicted_mean - 2 * pred.se_mean,
            pred.predicted_mean + 2 * pred.se_mean,
            **ci_kws,
        )
        if "lw" not in line_kws and "linewidth" not in line_kws:
            line_kws["lw"] = 2.0
        h = ax.plot(eval_x[:, 1], pred.predicted_mean, **line_kws)
    else:
        h = None

    # make the scatter plot
    scatter_kws = {} if scatter_kws is None else scatter_kws
    scatter_kws.setdefault("alpha", 0.8)
    if h is not None and "c" not in scatter_kws and "color" not in scatter_kws:
        scatter_kws.setdefault("c", h[0].get_color())
    ax.scatter(x, y, **scatter_kws)

    return fit_results
    

def _standardize_data(
    x: Union[None, str, pd.Series, Sequence] = None,
    y: Union[None, str, pd.Series, Sequence] = None,
    data: Optional[pd.DataFrame] = None,
    dropna: bool = True,
) -> Tuple[Sequence, Sequence]:
    # trying to avoid copying as much as possible
    if data is not None:
        if dropna:
            # drop NaNs
            data = data.dropna()

        x = data[x].values
        y = data[y].values
    else:
        if dropna:
            # drop NaNs
            x = np.asarray(x)
            y = np.asarray(y)

            mask = np.isnan(x) | np.isnan(y)
            x = x[~mask]
            y = y[~mask]

    return x, y
