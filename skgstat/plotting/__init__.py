import skgstat
from .variogram_plot import matplotlib_variogram_plot, plotly_variogram_plot
from .variogram_scattergram import matplotlib_variogram_scattergram, plotly_variogram_scattergram

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
