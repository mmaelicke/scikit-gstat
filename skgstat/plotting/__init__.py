import skgstat
from .variogram_plot import matplotlib_variogram_plot, plotly_variogram_plot

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

    else:
        skgstat.__backend__ = name
