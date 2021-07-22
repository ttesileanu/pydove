# if running in Jupyter with InlineBackend, ensure we don't crop or enlarge figures
try:
    from traitlets.config import get_config
    c = get_config()
    c.InlineBackend.print_figure_kwargs = {'bbox_inches': None}
except:
    pass

from pydove.regplot import regplot, scatter, fitplot, polyfit
from pydove.figure_manager import FigureManager
from pydove.color import colorbar, gradient_cmap
from pydove.plot import color_plot

__version__ = "0.3.5"
