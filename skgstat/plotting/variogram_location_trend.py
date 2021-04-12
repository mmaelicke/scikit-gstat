import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from itertools import cycle

try:
    import plotly
    import plotly.graph_objects as go
except ImportError:
    pass


def __get_trend(variogram, fig, **kwargs):
    # get the number of dimentsions
    N = variogram.coordinates.shape[1]

    # create the names
    if N <= 3:
        names = ['X', 'Y', 'Z'][:3]
    else:
        names = ['%d. dimension' % (_ + 1) for _ in range(N)]

    # cycle the default colors
    colors = cycle(plotly.colors.qualitative.Plotly)

    # only linear trend analysis supported:
    # TODO: this could be changed by kwargs...
    def model(x, m, b):
        return m * x + b

    for dim in range(N):
        x, y = (variogram.values, variogram.coordinates[:, dim])
        color = next(colors)

        # fit the model
        cof, cov = curve_fit(model, x, y)

        # apply the model
        xi = np.linspace(np.min(x), np.max(x), 100)
        yi = np.fromiter(map(lambda x: model(x, *cof), xi), dtype=float)

        # calculate R2
        y_star = np.fromiter(map(lambda x: model(x, *cof), x), dtype=float)
        r2 = 1 - (np.sum(np.power(y - y_star, 2)) / np.sum(np.power(y - np.mean(y), 2)))

        # add the trace
        fig.add_trace(
            go.Scatter(
                x=xi,
                y=yi,
                mode='lines+text',
                line=dict(dash='dash', width=0.7, color=color),
                name='%s trend' % names[dim],
                text=['y = %.2fx + %.2f [R²=%.2f]' % (cof[0], cof[1], r2) if i == 10 else '' for i, _ in enumerate(xi)],
                textfont_size=14,
                textposition='top center',
                textfont_color=color
            )
        )

    # after all traces are added, return
    return fig


def matplotlib_location_trend(variogram, axes=None, show=True, **kwargs):
    N = len(variogram.coordinates[0])

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
        axes.flatten()[i].plot([_[i] for _ in variogram.coordinates], variogram.values, '.r')
        axes.flatten()[i].set_xlabel('%d-dimension' % (i + 1))
        axes.flatten()[i].set_ylabel('value')

    # decrease margins
    plt.tight_layout()

    # show if needed
    if show:
        fig.show()

    return fig


def plotly_location_trend(variogram, fig=None, show=True, **kwargs):
    N = len(variogram.coordinates[0])
    if N <= 3:
        names = ['X', 'Y', 'Z'][:N]
    else:
        names = ['%d. dimension' % _ for _ in range(N)]

    # check if a figure is needed
    if fig is None:
        fig = go.Figure()

    x = variogram.values
    # switch to ScatterGL, if more than 5000 points will be plotted
    if len(x) * N >= 5000:
        GoCls = go.Scattergl
    else:
        GoCls = go.Scatter

    # plot
    for i in range(N):
        y = variogram.coordinates[:, i]
        fig.add_trace(
            GoCls(x=x, y=y, mode='markers', name=names[i])
        )

    fig.update_xaxes(title_text='Value')
    fig.update_yaxes(title_text='Coordinate dimension')

    # check if add_trend_line is given
    if kwargs.get('add_trend_line', False):
        fig = __get_trend(variogram, fig, **kwargs)

    # show figure if needed
    if show:
        fig.show()

    return fig
