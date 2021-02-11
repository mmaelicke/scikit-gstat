import numpy as np
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
except ImportError:
    pass


def __calculate_plot_data(variogram):
    tails = []
    heads = []

    sq_lags = squareform(variogram.lag_groups())

    for h in np.unique(variogram.lag_groups()):
        # get head and tail
        x, y = np.where(sq_lags == h)

        # add
        tails.append(variogram.values[x].flatten())
        heads.append(variogram.values[y].flatten())

    return tails, heads


def matplotlib_variogram_scattergram(variogram, ax=None, show=True, single_color=True, **kwargs):
    # get the plot data
    tails, heads = __calculate_plot_data(variogram)

    # create a new figure or use the given
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    # some settings
    color = 'orange' if single_color else None

    # plot
    h = np.concatenate(heads).ravel()
    t = np.concatenate(tails).ravel()
    ax.vlines(np.nanmean(t), np.min(t), np.nanmax(t), linestyles='--', color='red', lw=kwargs.get('lw', 1.5))
    ax.hlines(np.nanmean(h), np.nanmin(h), np.nanmax(h), linestyles='--', color='red', lw=kwargs.get('lw', 1.5))

    # plot
    for tail, head in zip(tails, heads):
        ax.scatter(tail, head, kwargs.get('size', 8), marker='o', color=color)

    # annotate
    ax.set_ylabel('head')
    ax.set_xlabel('tail')

    # show the figure
    if show:  # pragma: no cover
        fig.show()

    return fig


def plotly_variogram_scattergram(variogram, fig=None, show=False, single_color=False, **kwargs):
    # get the plot data
    tails, heads = __calculate_plot_data(variogram)

    # create a new Figure if needed
    if fig is None:
        fig = go.Figure()

    # some arguments
    lw = kwargs.get('line_width', kwargs.get('lw', 1.5))
    ld = kwargs.get('line_dash', 'dash')
    color = 'orange' if single_color else None

    # add vertical and horizontal lines
    try:
        h = np.concatenate(heads).ravel()
        t = np.concatenate(tails).ravel()
        fig.add_vline(x=np.nanmean(t), line_dash=ld, line_width=lw, line_color='red')
        fig.add_hline(y=np.nanmean(h), line_dash=ld, line_width=lw, line_color='red')
    except AttributeError:
        # add_hline and add_vline were added in plotly >= 4.12
        print("Can't plot lines, consider updating your plotly to >= 4.12")
        pass

    # do the plot
    for i, (tail, head) in enumerate(zip(tails, heads)):
        fig.add_trace(
            go.Scattergl(x=tail, y=head, mode='markers', marker=dict(size=kwargs.get('size', 4), color=color), name='Lag #%d' % i)
        )

    # add some titles
    fig.update_xaxes(title_text='Tail')
    fig.update_yaxes(title_text='Head')

    if single_color:
        fig.update_layout(showlegend=False)

    if show:
        fig.show()

    return fig
