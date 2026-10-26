import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import plotly.graph_objects as go
except ImportError:
    pass


TIME_FACTORS = {
    'seconds': 1.,
    'minutes': 60.,
    'hours': 3600.,
    'days': 86400.
}


def __calculate_plot_data(stvariogram, kind, **kwargs):
    # convert the time axis into the plotting unit
    factor = TIME_FACTORS.get(stvariogram.time_unit, 1.)
    xx, tt = stvariogram.meshbins
    tt = tt / factor
    z = stvariogram.experimental

    # the model is shown once fitted, unless the user decides otherwise
    show_model = kwargs.get('show_model', stvariogram.cof is not None)
    if not show_model or kind == 'matrix':
        return xx, tt, z, None, None, None

    # model spacing
    nx = kwargs.get('x_resolution', 100)
    nt = kwargs.get('t_resolution', 100)
    _xx, _tt = np.mgrid[
        0:np.nanmax(stvariogram.xbins):nx * 1j,
        0:np.nanmax(stvariogram.tbins):nt * 1j
    ]

    # apply the model
    lags = np.column_stack((_xx.flatten(), _tt.flatten()))
    _z = stvariogram.fitted_model(lags).reshape(_xx.shape)

    return xx, tt, z, _xx, _tt / factor, _z


def __time_label(stvariogram):
    if stvariogram.time_unit is None:
        return 'time lag'
    return 'time lag [%s]' % stvariogram.time_unit


def matplotlib_spatiotemporal_plot(stvariogram, kind='scatter', ax=None, elev=30, azim=220, **kwargs):
    # get the data, spanned over a bin meshgrid
    xx, tt, z, _xx, _tt, _z = __calculate_plot_data(stvariogram, kind, **kwargs)
    is_3d = kind in ('scatter', 'surface')

    # some settings
    c = kwargs.get('color', kwargs.get('c', 'b'))
    cmap = kwargs.get('cmap', 'RdYlBu_r')
    model_cmap = kwargs.get('model_color', 'terrain')
    alpha = kwargs.get('alpha', 0.8)

    # handle the axes
    if ax is not None:
        if is_3d and not isinstance(ax, Axes3D):
            raise ValueError('The passed ax object is not an instance of mpl_toolkis.mplot3d.Axes3D.')
        fig = ax.get_figure()
    else:
        fig = plt.figure(figsize=kwargs.get('figsize', (8, 6)))
        ax = fig.add_subplot(111, projection='3d' if is_3d else None)

    # do the plot
    if kind == 'scatter':
        ax.scatter(xx.flatten(), tt.flatten(), z.flatten(), c=c, depthshade=kwargs.get('depthshade', False))
    elif kind == 'surface':
        ax.plot_surface(xx, tt, z, cmap=cmap, alpha=alpha)
    elif kind in ('contour', 'contourf'):
        # draw the model if available, else the experimental variogram
        data = (_xx, _tt, _z) if _z is not None else (xx, tt, z)
        if kind == 'contour':
            C = ax.contour(*data, levels=kwargs.get('levels', 10), colors=c, linewidths=kwargs.get('linewidths', 0.3))
        else:
            C = ax.contourf(*data, levels=kwargs.get('levels', 10), cmap=cmap)
    elif kind == 'matrix':
        C = ax.imshow(
            z.T,
            origin='lower',
            aspect='auto',
            extent=(0, xx[-1, 0], 0, tt[0, -1]),
            cmap=cmap
        )
    else:
        raise ValueError('%s is not a valid plot' % kind)

    # add the model to 3D plots
    if is_3d:
        if _z is not None:
            ax.plot_surface(_xx, _tt, _z, cmap=model_cmap, alpha=kwargs.get('model_alpha', 0.5), linewidth=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_zlabel(kwargs.get('zlabel', 'semivariance [%s]' % stvariogram.estimator.__name__))
    elif kind in ('contourf', 'matrix') and kwargs.get('colorbar', True):
        plt.colorbar(C, ax=ax)

    # labels
    ax.set_xlabel(kwargs.get('xlabel', 'space lag'))
    ax.set_ylabel(kwargs.get('ylabel', __time_label(stvariogram)))

    return fig


def plotly_spatiotemporal_plot(stvariogram, kind='scatter', fig=None, **kwargs):
    # get the data, spanned over a bin meshgrid
    xx, tt, z, _xx, _tt, _z = __calculate_plot_data(stvariogram, kind, **kwargs)

    # get some settings
    c = kwargs.get('color', kwargs.get('c', 'black'))
    colorscale = kwargs.get('colorscale', 'Earth_r')
    model_colorscale = kwargs.get('model_color', 'Electric')
    alpha = kwargs.get('opacity', kwargs.get('alpha', 0.6))

    # handle the figure
    if fig is None:
        fig = go.Figure()

    # do the plot
    if kind == 'scatter':
        fig.add_trace(
            go.Scatter3d(
                x=xx.flatten(),
                y=tt.flatten(),
                z=z.flatten(),
                mode='markers',
                marker=dict(color=c, size=kwargs.get('size', 4)),
                name='experimental variogram'
            )
        )
    elif kind == 'surface':
        fig.add_trace(
            go.Surface(x=xx, y=tt, z=z, colorscale=colorscale, opacity=alpha, name='experimental variogram')
        )
    elif kind in ('contour', 'contourf'):
        # draw the model if available, else the experimental variogram
        x, t, data = (_xx, _tt, _z) if _z is not None else (xx, tt, z)
        fig.add_trace(
            go.Contour(
                x=x[:, 0],
                y=t[0, :],
                z=data.T,
                colorscale=colorscale,
                contours=dict(
                    coloring='lines' if kind == 'contour' else 'heatmap',
                    showlabels=kwargs.get('showlabels', True)
                )
            )
        )
    elif kind == 'matrix':
        fig.add_trace(
            go.Heatmap(x=xx[:, 0], y=tt[0, :], z=z.T, colorscale=colorscale)
        )
    else:
        raise ValueError('%s is not a valid plot' % kind)

    # add the model to 3D plots
    if kind in ('scatter', 'surface'):
        if _z is not None:
            fig.add_trace(
                go.Surface(x=_xx, y=_tt, z=_z, colorscale=model_colorscale, opacity=alpha, name='%s model' % stvariogram._model_name)
            )
        fig.update_layout(scene=dict(
            xaxis_title=kwargs.get('xlabel', 'space lag'),
            yaxis_title=kwargs.get('ylabel', __time_label(stvariogram)),
            zaxis_title=kwargs.get('zlabel', 'semivariance [%s]' % stvariogram.estimator.__name__)
        ))
    else:
        fig.update_layout(
            xaxis_title=kwargs.get('xlabel', 'space lag'),
            yaxis_title=kwargs.get('ylabel', __time_label(stvariogram))
        )

    return fig
