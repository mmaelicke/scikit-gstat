import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial.distance import squareform

try:
    import plotly.graph_objects as go
except ImportError:
    pass


def __calculate_plot_data(variogram, points):
    # get the direction mask
    direction_mask = squareform(variogram._direction_mask())

    # build a coordinate meshgrid
    n = len(variogram.coordinates)
    r = np.arange(n)
    x1, x2 = np.meshgrid(r, r)

    # handle the point pairs
    if isinstance(points, int):
        points = [points]
    if isinstance(points, (list, tuple)):
        point_mask = np.zeros((n, n), dtype=bool)
        point_mask[:, points] = True
    else:
        # use all points
        point_mask = np.ones((n, n), dtype=bool)

    start = variogram.coordinates[x1[direction_mask & point_mask]]
    end = variogram.coordinates[x2[direction_mask & point_mask]]

    # extract all lines
    lines = np.column_stack((
        start.reshape(len(start), 1, 2),
        end.reshape(len(end), 1, 2)
    ))

    return lines


def matplotlib_pair_field(
    variogram, ax=None,
    cmap='gist_rainbow',
    points='all',
    add_points=True,
    alpha=0.3,
    **kwargs
):
    # get the plot data
    lines = __calculate_plot_data(variogram, points)

    # align the colors
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(lines)))
    colors[:, 3] = alpha

    # get the figure and ax object
    if ax is None:
        figsize = kwargs.get('figsize', (8, 8))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # plot
    lc = LineCollection(lines, colors=colors, linewidths=1)
    ax.add_collection(lc)

    # add coordinates
    if add_points:
        ax.scatter(variogram.coordinates[:, 0], variogram.coordinates[:, 1], 15, c='k')
        if isinstance(points, list):
            ax.scatter(
                variogram.coordinates[:, 0][points],
                variogram.coordinates[:, 1][points],
                25, c='r'
            )

    # finish plot
    ax.autoscale()
    ax.margins(0.1)

    return fig


def plotly_pair_field(
    variogram,
    fig=None,
    points='all',
    add_points=True,
    alpha=0.3,
    **kwargs
):
    # get the plot data
    lines = __calculate_plot_data(variogram, points)

    # create a figure if none is passed
    if fig is None:
        fig = go.Figure()

    # plot all requested networks
    for line in lines:
        fig.add_trace(
            go.Scatter(x=line[:, 0], y=line[:, 1], mode='lines', opacity=alpha)
        )

    # add the coordinates as well
    if add_points:
        x = variogram.coordinates[:, 0]
        y = variogram.coordinates[:, 1]
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode='markers',
                marker=dict(color='black', size=5),
                text=['Coord: #%d' % i for i in range(len(x))]
            )
        )
        if isinstance(points, (list, tuple)):
            fig.add_trace(
                go.Scatter(
                    x=x[points], y=y[points], mode='markers',
                    marker=dict(color='red', size=15),
                    text=['Coordinate: #%d' % p for p in points]
                )
            )

    # get rid of the legend
    fig.update_layout(showlegend=False)

    return fig
