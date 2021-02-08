import numpy as np
import matplotlib.pyplot as plt 

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go 
except ImportError:
    pass


def __calculate_plot_data(variogram):
    # get the parameters
    _bins = variogram.bins
    _exp = variogram.experimental
    x = np.linspace(0, np.nanmax(_bins), 100)

    # apply the model
    y = variogram.transform(x)

    # handle the relative experimental variogram
    if variogram.normalized:
        _bins /= np.nanmax(_bins)
        y /= np.max(_exp)
        _exp /= np.nanmax(_exp)
        x /= np.nanmax(x)

    return x, y, _bins, _exp


def matplotlib_variogram_plot(variogram, axes=None, grid=True, show=True, hist=True):
    # get the plotting data
    x, y, _bins, _exp = __calculate_plot_data(variogram)

    # do the plotting
    if axes is None:
        if hist:
            fig = plt.figure(figsize=(8, 5))
            ax1 = plt.subplot2grid((5, 1), (1, 0), rowspan=4)
            ax2 = plt.subplot2grid((5, 1), (0, 0), sharex=ax1)
            fig.subplots_adjust(hspace=0)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
            ax2 = None
    elif isinstance(axes, (list, tuple, np.ndarray)):
        ax1, ax2 = axes
        fig = ax1.get_figure()
    else:
        ax1 = axes
        ax2 = None
        fig = ax1.get_figure()

    # ------------------------
    # plot Variograms
    ax1.plot(_bins, _exp, '.b')
    ax1.plot(x, y, '-g')

    # ax limits
    if variogram.normalized:
        ax1.set_xlim([0, 1.05])
        ax1.set_ylim([0, 1.05])
    if grid:
        ax1.grid(False)
        ax1.vlines(_bins, *ax1.axes.get_ybound(), colors=(.85, .85, .85), linestyles='dashed')
        # annotation
        ax1.axes.set_ylabel('semivariance (%s)' % variogram._estimator.__name__)
        ax1.axes.set_xlabel('Lag (-)')

    # ------------------------
    # plot histogram
    if ax2 is not None and hist:
        # calc the histogram
        _count = np.fromiter(
            (g.size for g in variogram.lag_classes()), dtype=int
        )

        # set the sum of hist bar widths to 70% of the x-axis space
        w = (np.max(_bins) * 0.7) / len(_count)

        # plot
        ax2.bar(_bins, _count, width=w, align='center', color='red')

        # adjust
        plt.setp(ax2.axes.get_xticklabels(), visible=False)
        ax2.axes.set_yticks(ax2.axes.get_yticks()[1:])

        # need a grid?
        if grid:  # pragma: no cover
            ax2.grid(False)
            ax2.vlines(_bins, *ax2.axes.get_ybound(), colors=(.85, .85, .85), linestyles='dashed')

        # anotate
        ax2.axes.set_ylabel('N')

    # show the figure
    if show:  # pragma: no cover
        fig.show()

    return fig


def plotly_variogram_plot(variogram, fig=None, grid=True, show=True, hist=True):
    # get the plotting data
    x, y, _bins, _exp = __calculate_plot_data(variogram)
    
    # create the figure
    if fig is None:
        if hist:
            fig = make_subplots(
                rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.0,
                specs=[
                    [{}], [{'rowspan': 4}], [None], [None], [None]
                ]
            )
        else:
            fig = make_subplots(rows=1, cols=1)
    elif isinstance(fig, go.Figure):
        pass
    else:
        raise ValueError('axes has to be None or a plotly.Figure.')

    # main plot
    fig.add_trace(
        go.Scatter(x=_bins, y=_exp, mode='markers',
                   marker=dict(color='blue'), name='Experimental'),
        row=2 if hist else 1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=y, mode='lines', marker=dict(color='green'),
            name='%s model' % variogram.model.__name__),
        row=2 if hist else 1, col=1
    )

    # update axis title
    fig.update_xaxes(title_text='Lag [-]', row=2 if hist else 1, col=1)
    fig.update_yaxes(
        title_text='semivariance (%s)' % variogram.estimator.__name__,
        row=2 if hist else 1, col=1
    )

    # hist
    if hist:
        # calculate
        _count = np.fromiter((g.size for g in variogram.lag_classes()), dtype=int)

        fig.add_trace(
            go.Bar(x=_bins, y=_count, marker=dict(color='red'), name='Histogram')
        )

        # title
        fig.update_yaxes(title_text='# of pairs', row=1, col=1)

    if show:
        fig.show()

    return fig
