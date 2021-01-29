""" Define plot-related functions. """

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from typing import Sequence, Optional


def color_plot(
    x: Sequence, y: Sequence, hue: Sequence, ax: Optional[plt.Axes] = None, **kwargs
) -> LineCollection:
    """ Make a line plot with variable hue.

    Note that this does not autoscale the axes by default. Either set axis limits
    manually or call `ax.autoscale()`.
    
    This function was adapted from
    
    https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html

    Parameters
    ----------
    x
        Horizontal component.
    y
        Vertical component.
    hue
        Color component. This is used to map into the colormap.
    ax
        Axes in which to draw.
    Any additional keyword arguments are directly passed to `LineCollection`.

    Returns the `LineCollection` that was drawn.
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, **kwargs)
    lc.set_array(np.asarray(hue))

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc
