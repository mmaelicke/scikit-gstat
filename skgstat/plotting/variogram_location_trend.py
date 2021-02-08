import numpy as np
import matplotlib.pyplot as plt

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
except ImportError:
    pass


def matplotlib_location_trend(variogram, axes=None, show=True):
    N = len(variogram._X[0])

    # create the figure
    if axes is None:
        # derive the needed amount of col and row
        nrow = int(round(np.sqrt(N)))
        ncol = int(np.ceil(N / nrow))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 6, nrow * 6))
    else:
        if not len(axes) == N:
            raise ValueError(
                'The amount of passed axes does not fit the coordinate' +
                ' dimensionality of %d' % N
            )
        fig = axes[0].get_figure()

    # plot
    for i in range(N):
        axes.flatten()[i].plot([_[i] for _ in variogram._X], variogram.values, '.r')
        axes.flatten()[i].set_xlabel('%d-dimension' % (i + 1))
        axes.flatten()[i].set_ylabel('value')

    # decrease margins
    plt.tight_layout()

    # show if needed
    if show:
        fig.show()

    return fig


def plotly_location_trend(variogram, fig=None, show=True):
    N = len(variogram._X[0])
    if N <= 3:
        names = ['X', 'Y', 'Z'][:N]
    else:
        names = ['%d. dimension' % _ for _ in range(N)]

    # check if a figure is needed
    if fig is None:
        fig = make_subplots(rows=1, cols=1)

    x = variogram.values
    # switch to ScatterGL, if more than 5000 points will be plotted
    if len(x) * N >= 5000:
        GoCls = go.Scattergl
    else:
        GoCls = go.Scatter

    # plot
    for i in range(N):
        y = variogram._X[:, i]
        fig.add_trace(
            GoCls(x=x, y=y, mode='markers', name=names[i]),
            row=1, col=1
        )

    fig.update_xaxes(title_text='Value')
    fig.update_yaxes(title_text='Coordinate dimension')

    # show figure if needed
    if show:
        fig.show()

    return fig
