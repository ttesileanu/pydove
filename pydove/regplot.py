""" Define a function that combines a scatter plot with a regression curve. """


import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erfinv
from typing import Union, Optional, Sequence, Tuple, Callable
from matplotlib.collections import PathCollection
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
    x_ci_kws: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
) -> Optional[RegressionResults]:
    """ Version of Seaborn `regplot` that returns fit results.

    This uses `pydove.scatter` with `pydove.polyfit` and `pydove.fitplot`.
    
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
    x_ci_kws 
        Additional keyword arguments to pass to `plt.errorbar` for the error bars at
        each unique value of `x` when `x_estimator` is used.
    ax
        Axes object to draw the plot onto, otherwise uses `plt.gca()`.

    Returns the results from `pydove.polyfit`, or `None` if the data is empty.
    """
    # convert data to a standard form
    x, y = _standardize_data(x=x, y=y, data=data, dropna=dropna)

    # check whether there's anything to do
    if len(x) == 0:
        return

    # handle some defaults
    ax = plt.gca() if ax is None else ax

    # figure out what color to use (unless we already know)
    if color is None:
        (h,) = ax.plot([], [])
        color = h.get_color()
        h.remove()

    # make the scatter plot
    if scatter:
        # set up the keywords
        scatter_kws = {} if scatter_kws is None else scatter_kws
        scatter_kws.setdefault("alpha", 0.8)
        if "c" not in scatter_kws and "color" not in scatter_kws:
            scatter_kws["color"] = color
        if marker is not None:
            scatter_kws.setdefault("marker", marker)
        if label is not None:
            scatter_kws.setdefault("label", label)

        # plot
        _scatter(
            x=x,
            y=y,
            x_estimator=x_estimator,
            seed=seed,
            x_jitter=x_jitter,
            y_jitter=y_jitter,
            x_ci_kws=x_ci_kws,
            **scatter_kws,
            ax=ax,
        )

    # calculate the fit
    fit_results = polyfit(x=x, y=y, order=order, logx=logx, with_constant=True)

    # draw the fit line and confidence interval
    if fit_reg:
        # figure out where to draw the line
        if truncate:
            low_x, high_x = np.min(x), np.max(x)
        else:
            low_x, high_x = ax.get_xlim()

        # set up the keywords
        line_kws = {} if line_kws is None else line_kws
        if "c" not in line_kws and "color" not in line_kws:
            line_kws["color"] = color
        if label is not None and not scatter:
            line_kws["label"] = label

        # plot
        fitplot(
            fit_results,
            x_range=(low_x, high_x),
            logx=logx,
            ci=ci,
            n_points=n_points,
            ci_kws=ci_kws,
            ax=ax,
            **line_kws,
        )

    return fit_results


def scatter(
    x: Union[None, str, pd.Series, Sequence] = None,
    y: Union[None, str, pd.Series, Sequence] = None,
    data: Optional[pd.DataFrame] = None,
    dropna: bool = True,
    x_estimator: Optional[Callable[[Sequence], float]] = None,
    seed: Union[int, np.random.Generator, np.random.RandomState] = 0,
    x_jitter: float = 0,
    y_jitter: float = 0,
    x_ci_kws: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> PathCollection:
    """ Make a scatter plot with some Seaborn-like extensions.

    This function basically performs the scatter-plot half of `sns.regplot`.

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
    dropna
        Drop any observations in which either `x` or `y` is not-a-number.
    x_estimator
        Group samples with the same value of `x` and apply an estimator to collapse all
        of the corresponding `y`values to a single number. If `x_ci` is given, a
        confidence interval for each estimate is also calculated and drawn.
    seed
        Seed or random number generator for jitter.
    x_jitter
        Amount of uniform random noise to add to the `x` variable. This is added only
        for the scatter plot, not for calculating the fit line. It's most useful for
        discrete data.
    y_jitter
        Amount of uniform random noise to add to the `y` variable. This is added only
        for the scatter plot, not for calculating the fit line. It's most useful for
        discrete data.
    x_ci_kws 
        Additional keyword arguments to pass to `plt.errorbar` for the error bars at
        each unique value of `x` when `x_estimator` is used.
    ax
        Axes object to draw the plot onto, otherwise uses `plt.gca()`.
    Any additional keyword arguments are passed to `plt.scatter`.

    Returns the output from `plt.scatter`.
    """
    # handle some defaults
    ax = plt.gca() if ax is None else ax

    # figure out what color to use (unless we already know)
    if "c" not in kwargs and "color" not in kwargs:
        (h,) = ax.plot([], [])
        kwargs["color"] = h.get_color()
        h.remove()

    # standardize data
    x, y = _standardize_data(x, y, data, dropna=dropna)

    # prepare data (jitter, grouping)
    xs, ys, ys_err = _prepare_data(
        x, y, seed=seed, x_jitter=x_jitter, y_jitter=y_jitter, x_estimator=x_estimator
    )

    # make the scatter plot
    kwargs.setdefault("alpha", 0.8)
    h = ax.scatter(xs, ys, **kwargs)

    # draw the error bars, if they exist
    if ys_err is not None:
        x_ci_kws = {} if x_ci_kws is None else x_ci_kws
        # XXX this won't work with vector color
        if "c" in kwargs:
            color = kwargs["c"]
        else:
            # if this wasn't given explicitly, it was set above
            color = kwargs["color"]

        x_ci_kws.setdefault("color", color)
        x_ci_kws.setdefault("elinewidth", 2.0)

        ax.errorbar(xs, ys, yerr=ys_err, ls="none", **x_ci_kws)

    return h


# creating an alias because the `scatter` kwarg of regplot shadows this function...
_scatter = scatter


def fitplot(
    fit_results: RegressionResults,
    x_range: Optional[Tuple] = None,
    x: Optional[Sequence] = None,
    logx: bool = False,
    ci: Optional[float] = 95,
    n_points: int = 100,
    ci_kws: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
):
    """ Plot a polynomial regression curve.

    This uses the fitting results from `statsmodels`.

    Parameters
    ----------
    fit_results
        The results of a polynomial fit, in `statsmodels` format. That is, it is assumed
        that the exogenous variables in the model are powers of either `x` or `log(x)`
        (see the `logx` option below). Specifically:
            exog[:, i] = (x or log(x)) ** (i + (1 - has_constant)) .
        To infer the existence of a constant term, `fit_results.k_constant` is used. The
        method `fit_results.get_prediction` is used to calculate the fit line and
        confidence interval. If `x_range` and `x` are not provided, the range of
        x-values for evaluating the fit is obtained from `fit_results.model.exog`. The
        function uses `fit_results.params` to determine the order of the polynomial.
    x_range
        Tuple `(x_min, x_max)` indicating the range of x-values to use for the fit.
    x
        Sequence of values where to evaluate the fit line. When this is provided,
        `x_range` and `n_points` are ignored.
    logx
        If true, assumes that the linear regression involves powers of `log(x)` instead
        of `x`.
    ci
        Size of confidence interval to draw for the regression line (in percent). This
        will be drawn using translucent bands around the regression line. Set to `None`
        to avoid drawing the confidence interval.
    n_points
        Number of points to use for drawing the fit line and confidence interval.
    ci_kws
        Additional keyword arguments to pass to `plt.fillbetween` for the confidence
        interval.
    ax
        Axes object to draw the plot onto, otherwise uses `plt.gca()`.
    Additional keywords are passed to `plt.plot()`.
    """
    # figure out the order of the fit
    has_constant = fit_results.k_constant
    order = len(fit_results.params) - has_constant

    # figure out the x values to use
    if x is None:
        if x_range is None:
            # find range from the model itself
            exog = fit_results.model.exog
            x_orig = exog[:, has_constant]
            if not logx:
                x_range = (np.min(x_orig), np.max(x_orig))
            else:
                x_range = (np.exp(np.min(x_orig)), np.exp(np.max(x_orig)))

        x = np.linspace(x_range[0], x_range[1], n_points)

    # build the matrix of predictor variables
    exog_fit = _build_poly_exog(x, order, has_constant, logx=logx)

    # calculate the predictions
    pred = fit_results.get_prediction(exog_fit)
    mu = pred.predicted_mean
    if ci is not None:
        std = np.sqrt(pred.var_pred_mean)
        n_std = np.sqrt(2) * erfinv(ci / 100)
        err = n_std * std
    else:
        err = None

    # set up keywords for fit line
    if "lw" not in kwargs and "linewidth" not in kwargs:
        kwargs["lw"] = 2.0

    # draw the fit line
    (h,) = ax.plot(x, mu, **kwargs)
    color = h.get_color()

    # set up keywords for confidence interval
    ci_kws = {} if ci_kws is None else ci_kws
    ci_kws.setdefault("alpha", 0.15)
    if "ec" not in ci_kws and "edgecolor" not in ci_kws:
        ci_kws["ec"] = "none"

    # use same color for confidence interval as for fit line (unless overridden)
    if not any(_ in ci_kws for _ in ["c", "color", "facecolor", "facecolors", "fc"]):
        ci_kws["facecolor"] = color

    # draw confidence interval, if available
    if err is not None:
        ax.fill_between(x, mu - err, mu + err, **ci_kws)


def polyfit(
    x: Union[None, str, pd.Series, Sequence] = None,
    y: Union[None, str, pd.Series, Sequence] = None,
    data: Optional[pd.DataFrame] = None,
    dropna: bool = True,
    order: int = 1,
    logx: bool = False,
    with_constant: bool = True,
) -> RegressionResults:
    """ Perform a polynomial fit using `statsmodels`.

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
    dropna
        Drop any observations in which either `x` or `y` is not-a-number.
    order
        If `order` is greater than 1, perform a polynomial regression.
    logx
        If true, estimate a linear regression of the form y ~ log(x), but plot the
        scatter plot and regression model in the input space. Note that `x` must be
        positive for this to work.
    with_constant
        If true, include a constant term in the fit.

    Returns a `statsmodels` regression results object. A `ValueError` exception is
    raised if the length of the data is zero.
    """
    # standardize data
    x, y = _standardize_data(x, y, data, dropna=dropna)

    # ensure we have data
    if len(x) == 0:
        raise ValueError("Data is empty, cannot fit.")

    # perform the fit
    exog = _build_poly_exog(x, order, has_constant=with_constant, logx=logx)
    fit_results = sm.OLS(y, exog).fit()

    return fit_results


def _standardize_data(
    x: Union[None, str, pd.Series, Sequence] = None,
    y: Union[None, str, pd.Series, Sequence] = None,
    data: Optional[pd.DataFrame] = None,
    dropna: bool = True,
) -> Tuple[Sequence, Sequence]:
    # handle dataframe vs. sequence inputs, trying to avoid copying as much as possible
    if isinstance(x, str):
        x = data[x]
    if isinstance(y, str):
        y = data[y]

    # basic length check
    if len(x) != len(y):
        raise ValueError("Different length x and y..")

    # drop invalid values if asked to
    if dropna:
        x = np.asarray(x)
        y = np.asarray(y)

        mask = ~(pd.isnull(x) | pd.isnull(y))

        x = x[mask]
        y = y[mask]

    return x, y


def _prepare_data(
    x: Sequence,
    y: Sequence,
    seed: Union[int, np.random.Generator, np.random.RandomState],
    x_jitter: float,
    y_jitter: float,
    x_estimator: Optional[Callable[[Sequence], float]],
) -> Tuple[Sequence, Sequence, Optional[Sequence]]:
    # add jitter, if asked to
    if x_estimator is None and (x_jitter != 0 or y_jitter != 0):
        if not hasattr(seed, "uniform"):
            rng = np.random.default_rng(seed)
        else:
            rng = seed

        if x_jitter != 0:
            x = np.asarray(x) + rng.uniform(-x_jitter, x_jitter, size=len(x))
        if y_jitter != 0:
            y = np.asarray(y) + rng.uniform(-y_jitter, y_jitter, size=len(y))

    # summarize data according to x_estimator, if asked to
    if x_estimator is not None:
        xs, xs_idxs = np.unique(x, return_index=True)
        ygrps = np.split(y, xs_idxs[1:])
        ys = [x_estimator(y_grp) for y_grp in ygrps]
        ys_err = [np.std(y_grp) for y_grp in ygrps]
    else:
        xs = x
        ys = y
        ys_err = None

    return xs, ys, ys_err


def _build_poly_exog(
    x: Sequence, order: int, has_constant: Union[bool, int], logx: bool
) -> np.ndarray:
    # build the matrix of predictor variables
    exog1 = np.asarray(x) if not logx else np.log(x)
    exog = np.empty((len(x), order + has_constant))
    for k in range(order + has_constant):
        exog[:, k] = exog1 ** (k + (1 - has_constant))

    return exog
