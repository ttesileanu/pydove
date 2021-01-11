""" Define a function that combines a scatter plot with a regression curve. """


import copy

import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erfinv
from typing import Union, Optional, Sequence, Tuple, Callable
from statsmodels.regression.linear_model import RegressionResults


def regplot(
    x: Union[None, str, pd.Series, Sequence] = None,
    y: Union[None, str, pd.Series, Sequence] = None,
    data: Optional[pd.DataFrame] = None,
    x_estimator: Optional[Callable[[Sequence], float]] = None,
    scatter: bool = True,
    fit_reg: bool = True,
    ci: Optional[float] = 95,
    n_points: int = 100,
    seed: Union[int, np.random.Generator, np.random.RandomState] = 0,
    order: int = 1,
    logx: bool = False,
    truncate: bool = True,
    dropna: bool = True,
    label: Optional[str] = None,
    x_jitter: float = 0,
    y_jitter: float = 0,
    color: Optional = None,
    marker: Optional = "o",
    scatter_kws: Optional[dict] = None,
    line_kws: Optional[dict] = None,
    ci_kws: Optional[dict] = None,
    x_estimator_ci_kws: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
) -> Optional[RegressionResults]:
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
    x_estimator
        Group samples with the same value of `x` and apply an estimator to collapse all
        of the corresponding `y`values to a single number. If `x_ci` is given, a
        confidence interval for each estimate is also calculated and drawn.
    scatter
        If true, draw the scatter plot.
    fit_reg
        If true, calculate and draw a linear regression.
    ci
        Size of confidence interval to draw for the regression line (in percent). This
        will be drawn using translucent bands around the regression line. Set to `None`
        to avoid drawing the confidence interval.
    n_points
        Number of points to use for drawing the fit line and confidence interval.
    seed
        Seed or random number generator for jitter.
    order
        If `order` is greater than 1, perform a polynomial regression.
    logx
        If true, estimate a linear regression of the form y ~ log(x), but plot the
        scatter plot and regression model in the input space. Note that `x` must be
        positive for this to work.
    truncate
        If true, the regression line is bounded by the data limits. Otherwise it extends
        to the x-axis limits.
    dropna
        Drop any observations in which either `x` or `y` is not-a-number.
    label
        Label to apply to either the scatter plot or regression line (if `scatter` is
        false) for use in a legend.
    x_jitter
        Amount of uniform random noise to add to the `x` variable. This is added only
        for the scatter plot, not for calculating the fit line. It's most useful for
        discrete data.
    y_jitter
        Amount of uniform random noise to add to the `y` variable. This is added only
        for the scatter plot, not for calculating the fit line. It's most useful for
        discrete data.
    color
        Color to apply to all plot elements; will be superseded by colors passed in
        `scatter_kws` or `line_kws`.
    marker
        Marker to use for the scatter-plot glyphs.
    scatter_kws
        Additional keyword arguments to pass to `plt.scatter` for the scatter plot.
    line_kws
        Additional keyword arguments to pass to `plt.plot` for the fit line.
    ci_kws
        Additional keyword arguments to pass to `plt.fillbetween` for the confidence
        interval.
    x_estimator_ci_kws 
        Additional keyword arguments to pass to `plt.errorbar` for the error bars at
        each unique value of `x` when `x_estimator` is used.
    ax
        Axes object to draw the plot onto, otherwise uses `plt.gca()`.

    Returns the results from `sm.OLS.fit()`.
    """
    # convert data to a standard form
    x, y = _standardize_data(x=x, y=y, data=data, dropna=dropna)

    # check whether there's anything to do
    if len(x) == 0:
        return

    # handle some defaults
    ax = plt.gca() if ax is None else ax

    # calculate best-fit linear or polynomial model
    x_fit0 = x if not logx else np.log(x)
    if order != 1:
        x_fit = np.empty((len(x), order + 1))
        for k in range(order + 1):
            x_fit[:, k] = x_fit0 ** k
    else:
        x_fit = sm.add_constant(x_fit0)
    fit_results = sm.OLS(y, x_fit).fit()

    # figure out what color to use (unless we already know, or we don't draw anything)
    if color is None and (scatter or fit_reg):
        (h,) = ax.plot([], [])
        color = h.get_color()
        h.remove()

    # add jitter, if asked to
    if not hasattr(seed, "uniform"):
        rng = np.random.default_rng(seed)
    else:
        rng = seed

    # make the scatter plot
    if scatter:
        scatter_kws = {} if scatter_kws is None else scatter_kws
        scatter_kws.setdefault("alpha", 0.8)
        if "c" not in scatter_kws and "color" not in scatter_kws:
            scatter_kws.setdefault("c", color)
        if marker is not None:
            scatter_kws.setdefault("marker", marker)
        if label is not None:
            scatter_kws.setdefault("label", label)

        if x_estimator is None:
            if x_jitter != 0:
                x = np.asarray(x) + rng.uniform(-x_jitter, x_jitter, size=len(x))
            if y_jitter != 0:
                y = np.asarray(y) + rng.uniform(-y_jitter, y_jitter, size=len(y))
            ax.scatter(x, y, **scatter_kws)
        else:
            # summarize data according to x_estimator
            xs, xs_idxs = np.unique(x, return_index=True)
            ygrps = np.split(y, xs_idxs[1:])
            ys = [x_estimator(y_grp) for y_grp in ygrps]
            ys_err = [np.std(y_grp) for y_grp in ygrps]

            if x_estimator_ci_kws is None:
                x_estimator_ci_kws = {}
            x_estimator_ci_kws.setdefault(
                "color", scatter_kws.get("c", scatter_kws.get("color", None))
            )
            x_estimator_ci_kws.setdefault("elinewidth", 2.0)

            ax.errorbar(xs, ys, yerr=ys_err, ls="none", **x_estimator_ci_kws)
            ax.scatter(xs, ys, **scatter_kws)

    # draw the fit line and confidence interval
    if fit_reg and len(x) > 2:
        if truncate:
            low_x = np.min(x)
            high_x = np.max(x)
        else:
            low_x, high_x = ax.get_xlim()

        eval_x = np.linspace(low_x, high_x, n_points)
        eval_x_fit0 = eval_x if not logx else np.log(eval_x)
        if order != 1:
            eval_x_fit = np.empty((len(x), order + 1))
            for k in range(order + 1):
                eval_x_fit[:, k] = eval_x_fit0 ** k
        else:
            eval_x_fit = sm.add_constant(eval_x_fit0)
        pred = fit_results.get_prediction(eval_x_fit)

        # set up keywords for fit line and error interval
        ci_kws = {} if ci_kws is None else ci_kws
        line_kws = {} if line_kws is None else line_kws

        if "c" not in line_kws and "color" not in line_kws:
            line_kws["color"] = color

        ci_kws.setdefault("alpha", 0.15)
        if "ec" not in ci_kws and "edgecolor" not in ci_kws:
            ci_kws["ec"] = "none"

        # if color is provided in line_kws, use same one in ci_kws (unless overridden)
        color_key = None
        if not any(
            _ in ci_kws for _ in ["c", "color", "facecolor", "facecolors", "fc"]
        ):
            if "c" in line_kws:
                color_key = "c"
            elif "color" in line_kws:
                color_key = "color"

            if color_key is not None:
                ci_kws["facecolor"] = line_kws[color_key]

        # draw the fit line and error interval
        mu = pred.predicted_mean
        std = np.sqrt(pred.var_pred_mean)

        # find out what multiple of the standard deviation to use (if any)
        if ci is not None:
            n_std = np.sqrt(2) * erfinv(ci / 100)
            err = n_std * std
            ax.fill_between(eval_x, mu - err, mu + err, **ci_kws)

        if "lw" not in line_kws and "linewidth" not in line_kws:
            line_kws["lw"] = 2.0
        if label is not None and not scatter:
            line_kws.setdefault("label", label)
        ax.plot(eval_x, pred.predicted_mean, **line_kws)

    return fit_results


def _standardize_data(
    x: Union[None, str, pd.Series, Sequence] = None,
    y: Union[None, str, pd.Series, Sequence] = None,
    data: Optional[pd.DataFrame] = None,
    dropna: bool = True,
) -> Tuple[Sequence, Sequence]:
    # basic length check
    if len(x) != len(y):
        raise ValueError("Different length x and y..")

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
