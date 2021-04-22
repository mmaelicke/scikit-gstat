import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import plotly.graph_objects as go
except ImportError:
    pass


def __calculate_plot_data(stvariogram, **kwargs):
    xx, yy = stvariogram.meshbins
    z = stvariogram.experimental
#    x = xx.flatten()
#    y = yy.flatten()

    # apply the model
    nx = kwargs.get('x_resolution', 100)
    nt = kwargs.get('t_resolution', 100)

    # model spacing
    _xx, _yy = np.mgrid[
        0:np.nanmax(stvariogram.xbins):nx * 1j,
        0:np.nanmax(stvariogram.tbins):nt * 1j
    ]
    model = stvariogram.fitted_model
    lags = np.vstack((_xx.flatten(), _yy.flatten())).T
    # apply the model
    _z = model(lags)

    return xx.T, yy.T, z, _xx, _yy, _z


def matplotlib_plot_3d(stvariogram, kind='scatter', ax=None, elev=30, azim=220, **kwargs):
    # get the data, spanned over a bin meshgrid
    xx, yy, z, _xx, _yy, _z = __calculate_plot_data(stvariogram, **kwargs)
    x = xx.flatten()
    y = yy.flatten()

    # some settings
    c = kwargs.get('color', kwargs.get('c', 'b'))
    cmap = kwargs.get('model_color', kwargs.get('cmap', 'terrain'))
    alpha = kwargs.get('alpha', 0.8)
    depthshade = kwargs.get('depthshade', False)

    # handle the axes
    if ax is not None:
        if not isinstance(ax, Axes3D):
            raise ValueError('The passed ax object is not an instance of mpl_toolkis.mplot3d.Axes3D.')
        fig = ax.get_figure()
    else:
        fig = plt.figure(figsize=kwargs.get('figsize', (10, 10)))
        ax = fig.add_subplot(111, projection='3d')

    # do the plot
    ax.view_init(elev=elev, azim=azim)
    if kind == 'surf':
        ax.plot_trisurf(x, y, z, color=c, alpha=alpha)
    elif kind == 'scatter':
        ax.scatter(x, y, z, c=c, depthshade=depthshade)
    else:
        raise ValueError('%s is not a valid 3D plot' % kind)


    # add the model
    if not kwargs.get('no_model', False):
        ax.plot_trisurf(_xx.flatten(), _yy.flatten(), _z, cmap=cmap, alpha=alpha)

    # labels:
    ax.set_xlabel('space')
    ax.set_ylabel('time')
    ax.set_zlabel('semivariance [%s]' % stvariogram.estimator.__name__)

    # return
    return fig


def plotly_plot_3d(stvariogram, kind='scatter', fig=None, **kwargs):
    # get the data spanned over a bin meshgrid
    xx, yy, z, _xx, _yy, _z = __calculate_plot_data(stvariogram, **kwargs)

    # get some settings
    c = kwargs.get('color', kwargs.get('c', 'black'))
    cmap = kwargs.get('model_color', kwargs.get('colorscale', kwargs.get('cmap', 'Electric')))
    alpha = kwargs.get('opacity', kwargs.get('alpha', 0.6))

    # handle the figue
    if fig is None:
        fig = go.Figure()

    # do the plot
    if kind == 'surf':
        fig.add_trace(
            go.Surface(
                x=xx,
                y=yy,
                z=z.reshape(xx.shape),
                opacity=0.8 * alpha,
                colorscale=[[0, c], [1, c]],
                name='experimental variogram'
            )
        )
    elif kind == 'scatter' or kwargs.get('add_points', False):
        fig.add_trace(
            go.Scatter3d(
                x=xx.flatten(),
                y=yy.flatten(),
                z=z,
                mode='markers',
                opacity=alpha,
                marker=dict(color=c, size=kwargs.get('size', 4)),
                name='experimental variogram'
            )
        )

    # add the model
    if not kwargs.get('no_model', False):
        fig.add_trace(
            go.Surface(
                x=_xx,
                y=_yy,
                z=_z.reshape(_xx.shape),
                opacity=max(1, alpha * 1.2),
                colorscale=cmap,
                name='%s model' % stvariogram.model.__name__
            )
        )

    # set some labels
    fig.update_layout(scene=dict(
        xaxis_title='space',
        yaxis_title='time',
        zaxis_title='semivariance [%s]' % stvariogram.estimator.__name__
    )) 

    # return
    return fig
