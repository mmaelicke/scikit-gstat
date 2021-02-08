import skgstat

from .variogram_plot import matplotlib_variogram_plot, plotly_variogram_plot
from .variogram_scattergram import matplotlib_variogram_scattergram, plotly_variogram_scattergram
from .variogram_location_trend import matplotlib_location_trend, plotly_location_trend
from .variogram_dd_plot import matplotlib_dd_plot, plotly_dd_plot
from .directtional_variogram import matplotlib_pair_field, plotly_pair_field
from .stvariogram_plot3d import matplotlib_plot_3d, plotly_plot_3d
from .stvariogram_plot2d import matplotlib_plot_2d, plotly_plot_2d
from .stvariogram_marginal import matplotlib_marginal, plotly_marginal


ALLOWED_BACKENDS = [
    'matplotlib',
    'plotly'
]


def backend(name=None):
    """
    """
    if name is None:
        return skgstat.__backend__

    elif name not in ALLOWED_BACKENDS:
        raise ValueError(
            "'%s' is not an allowed plotting backend.\nOptions are: [%s]" % 
            (name, ','.join(["'%s'" % _ for _ in ALLOWED_BACKENDS]))
        )

    elif name == 'plotly':
        try:
            import plotly
        except ImportError:
            print('You need to install plotly >=4.12.0 separatly:\npip install plotly')
            return

    # were are good to set the new backend
    skgstat.__backend__ = name
