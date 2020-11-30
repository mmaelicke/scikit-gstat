import numpy as np
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt 

try:
    import plotly.graph_objects as go 
except ImportError:
    pass

def __calculate_plot_data(variogram):
    tail = np.empty(0)
    head = tail.copy() 

    sq_lags = squareform(variogram.lag_groups())

    for h in np.unique(variogram.lag_groups()):
        # get head and tail
        x, y = np.where(sq_lags == h)

        # add
        tail = np.concatenate((tail, variogram.values[x]))
        head = np.concatenate((head, variogram.values[y]))

    return tail, head


def matplotlib_variogram_scattergram(variogram, ax=None, show=True):
    # get the plot data
    tail, head = __calculate_plot_data(variogram)

    # create a new figure or use the given
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    # plot
    ax.vlines(np.mean(tail), np.min(tail), np.max(tail), linestyles='--',
        color='red', lw=2)
    ax.hlines(np.mean(head), np.min(head), np.max(head), linestyles='--',
        color='red', lw=2)
    # plot
    ax.scatter(tail, head, 10, marker='o', color='orange')

    # annotate
    ax.set_ylabel('head')
    ax.set_xlabel('tail')

    # show the figure
    if show:  # pragma: no cover
        fig.show()

    return fig


def plotly_variogram_scattergram(variogram, fig=None, show=True):
    # get the plot data
    tail, head = __calculate_plot_data(variogram)

    # create a new Figure if needed
    if fig is None:
        fig = go.Figure()

    # add vertical and horizontal lines
    try:
        fig.add_vline(x=np.mean(tail), line_dash='dash', line_width=2, line_color='red')
        fig.add_hline(y=np.mean(head), line_dash='dash', line_width=2, line_color='red')
    except AttributeError:
        # add_hline and add_vline were added in plotly >= 4.12
        print("Can't plot lines, consider updating your plotly to >= 4.12")
        pass

    # do the plot
    fig.add_trace(
        go.Scattergl(x=tail, y=head, mode='markers', marker=dict(color='orange'))
    )

    # add some titles
    fig.update_xaxes(title_text='Tail')
    fig.update_yaxes(title_text='Head')

    if show:
        fig.show()

    return fig