import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
from scipy.interpolate import griddata

try:
    import plotly.graph_objects as go
except ImportError:
    pass


def matplotlib_plot_2d(stvariogram, kind='contour', ax=None, zoom_factor=100., levels=10, method='fast', **kwargs):
    # get or create the figure
    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', (8, 8)))

    # prepare the meshgrid
    xx, yy = stvariogram.meshbins
    z = stvariogram.experimental.T
    x = xx.flatten()
    y = yy.flatten()

    xxi = zoom(xx, zoom_factor, order=1)
    yyi = zoom(yy, zoom_factor, order=1)

    # interpolation, either fast or precise
    if method.lower() == "fast":
        zi = zoom(z.reshape((stvariogram.t_lags, stvariogram.x_lags)), zoom_factor, order=1, prefilter=False)
    elif method.lower() == "precise":
        # zoom the meshgrid by linear interpolation
        # interpolate the semivariance
        zi = griddata((x, y), z, (xxi, yyi), method='linear')
    else:
        raise ValueError("method has to be one of ['fast', 'precise']")

    # get the bounds
    zmin = np.nanmin(zi)
    zmax = np.nanmax(zi)

    # get the plotting parameters
    lev = np.linspace(0, zmax, levels)
    c = kwargs.get('color', kwargs.get('c', 'k'))
    cmap = kwargs.get('cmap', 'RdYlBu_r')

    # plot
    if kind.lower() == 'contour':
        ax.contour(xxi, yyi, zi, colors=c, levels=lev, vmin=zmin * 1.1, vmax=zmax * 0.9, linewidths=kwargs.get('linewidths', 0.3))
    elif kind.lower() == 'contourf':
        C = ax.contourf(xxi, yyi, zi, cmap=cmap, levels=lev, vmin=zmin *1.1, vmax=zmax * 0.9)
        if kwargs.get('colorbar', True):
            plt.colorbar(C, ax=ax)
    else:
        raise ValueError("%s is not a valid 2D plot" % kind)

    # some labels
    ax.set_xlabel(kwargs.get('xlabel', 'space'))
    ax.set_ylabel(kwargs.get('ylabel', 'time'))
    ax.set_xlim(kwargs.get('xlim', (0, stvariogram.xbins[-1])))
    ax.set_ylim(kwargs.get('ylim', (0, stvariogram.tbins[-1])))

    return fig


def plotly_plot_2d(stvariogram, kind='contour', fig=None, **kwargs):
    # get base data
    x = stvariogram.xbins
    y = stvariogram.tbins
    z = stvariogram.experimental.reshape((len(x), len(y))).T

    # get settings
    showlabels = kwargs.get('showlabels', True)
    colorscale = kwargs.get('colorscale', 'Earth_r')
    smooth = kwargs.get('line_smoothing', 0.0)
    coloring = kwargs.get('coloring', 'heatmap')
    if kind == 'contour':
        coloring = 'lines'
        lw = kwargs.get('line_width', kwargs.get('lw', 2))
        label_color = kwargs.get('label_color', 'black')
    else:
        label_color = kwargs.get('label_color', 'white')
        lw = kwargs.get('line_width', kwargs.get('lw', .3))

    # get the figure
    if fig is None:
        fig = go.Figure()

    # do the plot
    fig.add_trace(
        go.Contour(
            x=x,
            y=y,
            z=z,
            line_smoothing=smooth,
            colorscale=colorscale,
            contours=dict(
                coloring=coloring,
                showlabels=showlabels,
                labelfont=dict(
                    color=label_color,
                    size=kwargs.get('label_size', 14)
                )
            ),
            line_width=lw,
            colorbar=dict(
                title=f"semivariance ({stvariogram.estimator.__name__})",
                titleside='right'
            )
        )
    )

    # update the labels
    fig.update_layout(scene=dict(
        xaxis_title=kwargs.get('xlabel', 'space'),
        yaxis_title=kwargs.get('ylabel', 'time')
    ))

    return fig
