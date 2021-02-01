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
    mask = squareform(variogram._direction_mask())

    # build a coordinate meshgrid
    r = np.arange(len(self._X))
    x1, x2 = np.meshgrid(r, r)
    start = self._X[x1[mask]]
    end = self._X[x2[mask]]

    # handle lesser points
    if isinstance(points, int):
        points = [points]
    if isinstance(points, list):
        _start, _end = list(), list()
        for p in self._X[points]:
            _start.extend(start[np.where(end == p)[0]])
            _end.extend(end[np.where(end == p)[0]])
        start = np.array(_start)
        end = np.array(_end)

        # extract all lines
        lines = np.column_stack((
            start.reshape(len(start), 1, 2), 
            end.reshape(len(end), 1, 2)
        ))

        return lines


def matplotlib_pair_field(variogram, ax=None, cmap='gist_rainbow', points='all', add_points=True, alpha=0.3, **kwargs):
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
        ax.scatter(variogram._X[:, 0], variogram._X[:, 1], 15, c='k')
        if isinstance(points, list):
            ax.scatter(variogram._X[:, 0][points], variogram._X[:, 1][points], 25, c='r')

    # finish plot
    ax.autoscale()
    ax.margins(0.1)

    return fig
