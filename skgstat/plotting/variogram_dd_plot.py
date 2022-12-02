import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

try:
    import plotly.graph_objects as go
except ImportError:
    pass


def __calculate_plot_data(variogram):
    # get all distances and residual diffs
    dist = variogram.distance
    diff = variogram.pairwise_diffs

    return diff, dist


def matplotlib_dd_plot(variogram, ax=None, plot_bins=True, show=True):
    # get the plotting data
    _diff, _dist = __calculate_plot_data(variogram)

    # create the plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.get_figure()

    # plot the bins
    if plot_bins:
        _bins = variogram.bins
        ax.vlines(_bins, 0, np.max(_diff), linestyle='--', lw=1, color='r')

    # plot
    ax.scatter(_dist, _diff, 8, color='b', marker='o', alpha=0.5)

    # set limits
    ax.set_ylim((0, np.max(_diff)))
    ax.set_xlim((0, np.max(_dist)))
    ax.set_xlabel('separating distance')
    ax.set_ylabel('pairwise difference')
    ax.set_title('Pairwise distance ~ difference')

    # show the plot
    if show:  # pragma: no cover
        fig.show()

    return fig


def plotly_dd_plot(variogram, fig=None, plot_bins=True, show=True):
    # get the plotting data
    _diff, _dist = __calculate_plot_data(variogram)

    # create a new Figure if needed
    if fig is None:
        fig = go.Figure()

    # plot
    fig.add_trace(
        go.Scattergl(
            x=_dist, y=_diff,
            mode='markers', marker=dict(color='blue', opacity=0.5)
        )
    )

    # plot the bins
    if plot_bins:
        for _bin in variogram.bins:
            fig.add_vline(x=_bin, line_dash='dash', line_color='red')

    # titles
    fig.update_layout(
        title='Pairwise distance ~ difference',
        xaxis_title='separating distance',
        yaxis_title='pairwise difference'
    )

    if show:
        fig.show()

    return fig
