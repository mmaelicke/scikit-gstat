"""
4.22 Plotting
============
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
Creative Commons BY 4.20 license. Please cite the original publications if you
use the data, and **not** SciKit-GStat.

References
^^^^^^^^^^

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
# 4.2.1 Directional Variogram
# ---------------------------
# Load a artificial random field, generated from a Gaussian covariance function,
# with a 2x larger range in x-axis direction:
ac, av = skg.data.aniso(N=300, seed=42).get('sample')

# %%
# Estimate the directional variogram with a few more lag classes and an azimuth
# of 90°. The tolerance is set rather low to illustrate the graphs better
# (fewer point connections.):
DV = skg.DirectionalVariogram(ac, av, n_lags=20, azimuth=40., tolerance=15.0)
print(DV)


# %%
# 4.2.2 Backend
# -------------
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

# %%
# 4.2.3 :func:`pair_field <skgstat.DirectionalVariogram.pair_field>`
# ------------------------------------------------------------------
# 
# The :class:`DirectionalVariogram <skgstat.DirectionalVariogram>` class is
# inheriting from :class:`Variogram <skgstat.Variogram>`. Therefore all plotting
# method shown above are available for directional variograms, as well.
# Additionally, there is one more plotting method,
# :func:`DirectionalVariogram.pair_field <skgstat.DirectionalVariogram.pair_field>`.
# This function will plot all coordinate locations and draw a line between all
# point pairs, that were not masked by the directional mask array and will, thus,
# be used for variogram estimation. By default, the method will draw all lines for
# all point pairs and you will see nothing on the plot. But there is also the
# possibility to draw these lines only for a subset of the coordinate locations.

# Matplotlib
# ^^^^^^^^^^^
backend('matplotlib')
fig = DV.pair_field()

# %%
# Obviously, one can see the ``azimuth`` (40°) and narrow ``tolerance`` (15°)
# settings in the cone-like shapes of the connection lines, but the whole plot
# is not really instructive or helpful like this. 
# Using the ``points`` keyword, you can show the lines only for a given set of
# coordinate locations. You have to pass a list of coordinate indices. With
# ``add_points=True``, the seleceted points will be highlighted in red.
fig = DV.pair_field(points=[0, 42, 104, 242], add_points=True)

# %%
# Plotly
# ^^^^^^
# 
# **Note:** It is not recommended to plot the full
# :func:`pair_field <skgstat.DirectionalVariogram.pair_field>` with all points
# using plotly. Due to the implementation, that makes the plot really,
# really slow for rendering.
backend('plotly')
fig = DV.pair_field(points=[0,42, 104, 242], add_points=True, show=False)
fig
