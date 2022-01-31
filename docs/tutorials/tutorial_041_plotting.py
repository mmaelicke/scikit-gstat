"""
4. Plotting
===========
At the core of SciKit-GStat is a set of classes, that can be used interactively
to perform variogram analysis. One important aspect of this analysis is a rich
collection of plotting functions. These are directly available as class methods
of the :class:`Variogram <skgstat.Variogram>`, 
:class:`DirectionalVariogram <skgstat.DirectionalVariogram>` and 
:class:`SpaceTimeVariogram <skgstat.SpaceTimeVariogram>` method.
With version ``0.3.3``, SciKit-GStat implements two different plotting backend: 
`matplotlib <https://matplotlib.org/>`_ and `plotly <https://plotly.com/python/>`_.
Generally speaking, matplotlib is great for creating publication ready figures
in a variety of formats, including vector-graphic PDF files. Plotly, on the other
hand, will translate the figure into their Javascript library and open a webbrowser
with an interactive plot. This way you can obtain the same figure either for
your publication as PDF, or as a HTML object that can be injected into a
project report website.

With the newly introduced :mod:`skgstat.plotting` backend, you can easily read
and change the backend with a single convenient function. The default backend
is matplotlib. Please be aware, that `plotly` is only a soft dependency,
meaning you need to take care of the installation yourself, to keep
SciKit-GStat's dependency list shorter.

The data used to create the :class:`Variogram <skgstat.Variogram>` and 
:class:`DirectionalVariogram <skgstat.DirectionalVariogram>` is from 
Mälicke (2021). Here, pancake dataset is used.
The spatio-temporal data is derived from Fersch et al. (2020). From that data
publication, the wireless sensor network data is used. The originaly published
15 minutes intervals soil temperature data at 20 cm depth was taken for all 55
stations and aggregated to mean hourly values. To further decrease the data size,
only every 6th data point is used here. Estimating the full data set will take
approx. 120GB RAM and processing took about 30 minutes. The results for the
thinned data sample are very comparable.

Both data samples can either be obtained by the orignial publications,
or from the SciKit-GStat documentation. Both samples are published under
Creative Commons BY 4.0 license. Please cite the original publications if you
use the data, and **not** SciKit-GStat.

**References**

Fersch, Benjamin, et al. "A dense network of cosmic-ray neutron sensors for soil moisture observation in a pre-alpine headwater catchment in Germany." Earth System Science Data Discussions 2020 (2020): 1-35.

Mälicke, M.: SciKit-GStat 1.0: A SciPy flavoured geostatistical variogram estimation toolbox written in Python, Geosci. Model Dev. Discuss. [preprint], https://doi.org/10.5194/gmd-2021-174, in review, 2021.
"""
import skgstat as skg
from skgstat.plotting import backend
import numpy as np
import json
import warnings
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# %%
# 4.1 Variogram
# -------------
# Load a pancake sample from the data directory.
c, v = skg.data.pancake(N=300, seed=42).get('sample')

# %%
# Estimate a variogram, with a few more lag classes, as there are enough observation points available.
V = skg.Variogram(c,v, n_lags=25)
print(V)

# %%
# 4.2 Backend
# -----------
#
# You can switch to `plotly` as a plotting backend by calling the 
# :mod:`plotting.backend` function and passing the name of the backend.
# Note that plotly is only a soft dependency and will not automatically be
# installed along with SciKit-GStat. You can install it like:
# 
# .. code-block:: bash
# 
#   pip install plotly
# 
# Note that in a Jupyter environment you might want to use the plotly.offline
# environment to embed the needed Javascript into the notebook. In these cases
# you have to catch the Figure object and use the iplot function from the
# offline submodule.
#
# 4.3 Variogram
# -------------
# 
# 4.3.1 :func:`Variogram.plot <skgstat.Variogram.plot>`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The :func:`Variogram.plot <skgstat.Variogram.plot>` is the main plotting
# function in SciKit-GStat.
# Before you use the variogram for further geostatistical methods, like kriging,
# or further analysis, make sure, that a suitable model was found and fitted
# to the experimental data. Further, you have to make sure that the statistical
# foundation of this estimation is sound, the lag classes are well designed and
# backed by a suiatable amount of data. 
# Otherwise, any other geostatistical analysis or method will have to fail,
# no matter how nice the results might look like.

# from skgstat.plotting import backend
backend('plotly')

# %%
# Plotly
# """"""
fig = V.plot(show=False)
fig

# %%
# A useful argument for ``plot`` is the ``ax``, this takes a 
# ``matplotlib.AxesSubplot`` for the ``'matplotlib'`` backend and a 
# ``plotly.Figure`` for the  ``'plotly'`` backend.
# You need to supply the correct amount of subplots (two). For convenience,
# the histogram in the upper subplot can be disabled.
fig = make_subplots(rows=1, cols=1)
fig.update_layout(
    width=800,
    height=200,
    template='seaborn',
    showlegend=False, 
    margin=dict(l=0, r=0, b=0, t=0)
)

V.plot(axes=fig, hist=False, show=False)
fig

# %%
# The :func:`Variogram.plot <skgstat.Variogram.plot>` functions is customizable
# and takes a lot of arguments. However, the same interface is used as for the
# ``matplotlib`` version of that function. Many matplotlib arguments are mapped
# to the corresponding plotly arguments. Beyond that, you can either try common
# plotly arguments, or update the figure afterwards:
fig = V.plot(show=False)

fig.update_layout(
    legend=dict(x=0.05, y=1.1, xanchor='left', yanchor='top', orientation='h'),
    template='plotly_dark',
    annotations=[dict(
        text="AWESOME",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        font=dict(color="white", size=100),
        textangle=-30,
        opacity=.3
    )]
)
fig

# %%
#  Matplotlib
#  """"""""""
backend('matplotlib')

fig = V.plot()

# %%
# With matplotlib, you can set any ``matplotlib.AxesSubplot`` as ``axes`` to
# plot on other figures. You can send two axes, for the variogram and the
# histogram, or only one and disable the histogram plotting.
fig, ax = plt.subplots(2,2, figsize=(8, 4))

fig = V.plot(axes=ax.flatten()[1], hist=False)

# %%
# 4.3.2 `Variogram.scattergram`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# You can plot a scattergram of all point pairs formed by the class.
# The pairs can be grouped by the lag classes, they were formed in. This way you
# can analyze how the two values of the point pait (head and tail) scatter and
# if this follows a pattern (i.e. anisotropy). It is recommended to use the
# ``'plotly'`` backend, as you can click on the legend entries to hide a
# specific class, or double-click to show only the selected lag class.
# This makes it much easier to inspect the classes.
# 
# Plotly
# """"""
backend('plotly')
fig = V.scattergram(show=False)
fig

# %%
# It is, however possible to re-create the plot that was used up to SciKit-GStat
# version ``0.3.0`` with only one color. This is still the default for the
# ``'matplotlib'`` backend. 
fig = V.scattergram(single_color=True, show=False)
fig

# %%
# Matplotlib
# """"""""""
# backend('matplotlib')
fig = V.scattergram()

# %%
# 4.3.3 `Variogram.location_trend`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Another useful helper plot is the
# :func:`location_trend <skgstat.Variogram.location_trend>`. This will plot the
# observation values related to their coordinate position, for each coordinate
# dimension separatedly. With the ``'plotly'`` backend, each dimension will appear
# as a coloured group in a single plot. By double-clicking the legend, you can
# inspect each group separately.
# 
# The ``'plotly'`` backend will automatically switch the used plot type from a
# ordinary scatter-plot to a WebGL backed scatter-plot, if there are more than
# 5000 observations. This will add some startup-overhead for the plot to appear,
# but the interactivity actions (like pan, zoom) are speed up by magnitudes.
# 
# Plotly
# ^^^^^^
backend('plotly')
fig = V.location_trend(show=False)
fig

# %%
# Since version ``0.3.5`` the :func:`location_trend <skgstat.Variogram.location_trend>`#
# function accepts a ``add_trend_line`` parameter, that defaults to ``False``.
# If set to true, the class will fit linear models to each of the point clouds
# and output a trend line. It will also calculate the R², which you can use to
# either accept the input data as trend free or not (a high R² indicates a
# **linear** trend and hence you should decline using the input data).
fig = V.location_trend(add_trend_line=True, show=False)
fig

# Matplotlib
# """"""""""
# 
# There is a difference between the ``'matplotlib'`` and ``'plotly'`` backend in
# this plotting function. As Plotly utilizes the legend by default to show and
# hide traces on the plot, the user can conveniently switch between the
# coordinate dimensions. 
# In Matplotlib, the figures are not interactive by default and therefore
# SciKit-GStat will create one subplot for each coordinate dimension.
backend('matplotlib')
fig = V.location_trend()

# %%
# 4.3.4 :func:`distance_difference plot <skgstat.Variogram.distance_difference_plot>`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# The final utility plot presented here is a scatter-plot that relates all
# pairwise-differences in value to the spatial distance of the respective point
# pairs. This can already be considered to be a variogram. For convenience, the
# plotting method will mark all upper lag class edges in the plot. This can already
# give you an idea, if the number of lag classes is chosen wisely, or if you need to
# adjust. To estimate valid, expressive variograms, this is maybe the most important
# preparation step. If your lag classes do not represent your data well, you will
# never find a useful variogram.
# 
# Plotly
# """"""
# 
backend('plotly')
fig = V.distance_difference_plot(show=False)
fig

# %%
# You might also consider to adapt the maximum lag distance using this plot, to
# exclude distances that are not well backed by data. Alternatively,
# the binning method can be changed. Or both
Vcopy = V.clone()
Vcopy.bin_func = 'uniform'

fig = Vcopy.distance_difference_plot(show=False)
fig


# Matplotlib
# """"""""""
backend('matplotlib')
fig = V.distance_difference_plot()