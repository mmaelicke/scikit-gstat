from typing import List

import numpy as np
import matplotlib.pyplot as plt

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
except ImportError:
    pass


def matplotlib_cv_matrix(
    variograms: List[List['Variogram']],
    add_model: bool = True,
    sharex: bool = True,
    sharey: bool = True
):
    """"""
    # get the dimension
    n_rows, n_cols = np.asarray(variograms).shape

    # create the figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_rows * 5, n_cols * 5), sharex=sharex, sharey=sharey)

    # plot each of the primary / cross variograms  
    for i in range(n_rows):
        for j in range(n_cols):
            # add the empirical variogram
            v = variograms[i][j]
            axes[i][j].plot(v.bins, v.experimental, 'b.')

            # add the model
            if add_model:
                x = np.linspace(v.bins[0], v.bins[-1], num=100)
                y = v.transform(x)
                axes[i][j].plot(x, y, 'g-')

    # return the figure
    return fig


def plotly_cv_matrix(
    variograms: List[List['Variogram']],
    add_model: bool = True,
    sharex: bool = True,
    sharey: bool = True
):
    """"""
    # get the dimension
    n_rows, n_cols = np.asarray(variograms).shape

    # build the figure
    fig = make_subplots(n_rows, n_cols, shared_xaxes=sharex, shared_yaxes=sharey)

    # plot each of the primary / cross variograms  
    for i in range(n_rows):
        for j in range(n_cols):
            # add the empirical variogram
            v = variograms[i][j]
            fig.add_trace(go.Scatter(
                x=v.bins,
                y=v.experimental,
                mode='markers',
                marker=dict(color='blue'),
                showlegend=False
            ), row = i + 1, col = j + 1)

            # add the model
            if add_model:
                x = np.linspace(v.bins[0], v.bins[-1], num=100)
                y = v.transform(x)
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    line=dict(color='green'),
                    showlegend=False
                ), row = i + 1, col = j + 1)

    # return the figure
    return fig

