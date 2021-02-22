import numpy as np
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    pass


def __calc_plot_data(stvariogram, **kwargs):
    # get the marginal experimental variograms
    vx = stvariogram.XMarginal.experimental
    vy = stvariogram.TMarginal.experimental

    res = kwargs.get('model_resolution', 100)

    # get the model
    xx = np.linspace(0, stvariogram.xbins[-1], res)
    xy = np.linspace(0, stvariogram.tbins[-1], res)
    y_vx = stvariogram.XMarginal.transform(xx)
    y_vy = stvariogram.TMarginal.transform(xy)

    return xx, xy, y_vx, y_vy


def matplotlib_marginal(stvariogram, axes=None, sharey=True, include_model=False, **kwargs):
    # check if an ax needs to be created
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=kwargs.get('figsize', (12, 6)),sharey=sharey)
    else:
        if len(axes) != 2:
            raise ValueError('axes needs to an array of two AxesSubplot objects')
        fig = axes[0].get_figure()

    # get some settings
    x_style = kwargs.get('x_style', 'ok' if include_model else '-ob')
    t_style = kwargs.get('t_style', 'ok' if include_model else '-og')

    # handle the twin axes
    ax = axes[0]
    ax2 = axes[1]
    ax3 = ax2.twinx()
    ax3.get_shared_y_axes().join(ax3, ax)

    # plot the marginal experimental variogram
    ax.plot(stvariogram.xbins, stvariogram.XMarginal.experimental, x_style)
    ax3.plot(stvariogram.tbins, stvariogram.TMarginal.experimental, t_style)

    if include_model:
        xx, xy, y_vx, y_vy = __calc_plot_data(stvariogram, **kwargs)

        # plot
        ax.plot(xx, y_vx, '-b')
        ax3.plot(xy, y_vy, '-g')

    # set labels
    ax.set_xlabel('distance [spatial]')
    ax.set_ylabel('semivariance [%s]' % stvariogram.estimator.__name__)
    ax2.set_xlabel('distance [temporal]')
    if not sharey:
        ax3.set_ylabel('semivariance [%s]' % stvariogram.estimator.__name__)

    # set title and grid
    ax.set_title('spatial marginal variogram')
    ax2.set_title('temporal marginal variogram')

    ax.grid(which='major')
    ax2.grid(which='major')
    plt.tight_layout()

    # return
    return fig


def plotly_marginal(stvariogram, fig=None, include_model=False, **kwargs):
    shared_yaxes = kwargs.get('sharey', kwargs.get('shared_yaxis', True))
    # check if a figure needs to be created
    if fig is None:
        fig = go.Figure()
    try:
        fig.set_subplots(rows=1, cols=2, shared_yaxes=shared_yaxes)
    except ValueError:
        # figure has alredy subplots
        pass

    # get some settings
    x_color = kwargs.get('x_color', 'black' if include_model else 'green')
    t_color = kwargs.get('t_color', 'black' if include_model else 'blue')

    # plot the marginal experimental variogram
    fig.add_trace(
        go.Scatter(
            name='Spatial marginal variogram',
            x=stvariogram.xbins,
            y=stvariogram.XMarginal.experimental,
            mode='markers' if include_model else 'markers+lines',
            marker=dict(color=x_color),
            line=dict(color=x_color),
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            name='Temporal marginal variogram',
            x=stvariogram.tbins,
            y=stvariogram.TMarginal.experimental,
            mode='markers' if include_model else 'markers+lines',
            marker=dict(color=t_color),
            line=dict(color=t_color)
        ), row=1, col=2
    )

    # handle models
    if include_model:
        xx, yy, y_vx, y_vy = __calc_plot_data(stvariogram, **kwargs)

        # add the models
        fig.add_trace(
            go.Scatter(
                name='spatial %s model' % stvariogram.XMarginal.model.__name__,
                x=xx,
                y=y_vx,
                mode='lines',
                line=dict(color='blue')
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                name='temporal %s model' % stvariogram.TMarginal.model.__name__,
                x=yy,
                y=y_vy,
                mode='lines',
                line=dict(color='green')
            ), row=1, col=2
        )

    # update the layout
    fig.update_xaxes(title_text='distance [spatial]', row=1, col=1)
    fig.update_xaxes(title_text='distance [temporal]', row=1, col=2)
    fig.update_yaxes(title_text='semivariance [%s]' % stvariogram.estimator.__name__, row=1, col=1)
    if not shared_yaxes:
        fig.update_yaxes(title_text='semivariance [%s]' % stvariogram.estimator.__name__, row=1, col=2)

    fig.update_layout(
        legend=dict(
            orientation='h',
            x=0,
            y=1.05,
            xanchor='left',
            yanchor='bottom'
        )
    )

    return fig
