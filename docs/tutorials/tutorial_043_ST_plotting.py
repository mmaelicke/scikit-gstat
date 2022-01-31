"""
4.3 Plotting
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
Creative Commons BY 4.0 license. Please cite the original publications if you
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
# 4.3.1 Load Data
# ---------------
# %%
# Load the TERENO soil temperature data from Fersch et al. (2020):
with open('./data/tereno_fendt/tereno.json', 'r') as js:
    data_obj = json.load(js)

coords = np.array(data_obj['coordinates'])
vals = np.array(data_obj['values'])
print(data_obj['description'])

# %%
# Estimate the spatio-temporal variogram with a product-sum model.
# Only every 6th hour is taken into account to decrease the memory footprint.
# If you use the full dataset, you need ^120 GiB RAM. 
# The marginal variograms are kept as they are.
STV = skg.SpaceTimeVariogram(coords, vals[:,::6], x_lags=20, t_lags=20, model='product-sum')
print(STV)

# %%
# 4.3.2 Backend
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
#

# %%
# 4.3.5 ST Variogram
# ------------------
# The :class:`SpaceTimeVariogram <skgstat.SpaceTimeVariogram>` does not
# inherit from the :class:`Variogram <skgstat.Variogram>`class and thus,
# its plotting methods are not available for space time variograms.
# However, the :class:`SpaceTimeVariogram <skgstat.SpaceTimeVariogram>`
# has two properties, ``SpaceTimeVariogram.XMarginal`` and ``SpaceTimeVariogram.TMarginal``,
# which are both instances of :class:`Variogram <skgstat.Variogram>`
# for the spatial and temporal marginal variogram. These instances in turn,
# have all plotting methods available, in addition to the plotting methods of
# :class:`SpaceTimeVariogram <skgstat.SpaceTimeVariogram>` itself.

# 4.3.5.1 `plot(kind='scatter') <skgstat.SpaceTimeVariogram.plot>`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# The scatter plot can be used to inspect the experimental variogram data on a
# spatial and temporal axis, with the fitted spatio-temporal model fitted to the data.
# 
# Plotly
# """"""
backend('plotly')
fig = STV.plot(kind='scatter', show=False)
fig

# %%
# The method can also remove the model from the plot. This can be helpful in
# case the experimental data should be analyzed. Then, the model plane might be disturbing.
fig = STV.plot(kind='scatter', no_model=True, show=False)
fig

# %%
# And finally, the experimental point data can be connected to a surface grid,
# to emphasize an apparent structure more easily in a 3D plot. This can be done by switching to ``kind='surf'``.
fig = STV.plot(kind='surf', show=False)
fig


# Matplotlib
# """"""""""
backend('matplotlib')
fig = STV.plot(kind='surf')

# %%
# 4.3.5.2 :func:`contour <skgstat.SpaceTimeVariogram.contour>`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 3D plots are great for data exploration, especially if they are interactive.
# For publications, 3D plots are not that helpful. Additionally, it can be quite
# tricky sometimes to find a good angle to focus on the main message of a 3D plot. 
# Hence, there are more plotting modes. They can either be used by
# setting ``kind='contour'`` or ``kind='contourf'``. Alternatively, these two
# plotting types also have their own method.
# In both cases, the experimental semi-variance is plotted on a two dimensional
# plane. The spatial dimension can be found on the x-axis and the temporal
# dimension on the y-axis. The semi-variance itself is shown as a contour plot,
# that can either only plot the lines (``'contour'``) or filled areas for each
# contour (``'contourf'``).
# 
# Plotly
# """"""
backend('plotly')
fig = STV.contour(show=False)
fig

# %%
#
fig = STV.contourf(show=False)
fig

# %%
# Matplotlib
# """"""""""
# 
# The matplotlib versions of the contour plots are not that sophisticated,
# but the returned figure can  be adjusted to your needs.
backend('matplotlib')
fig = STV.plot(kind='contour')


# %%
# 
fig = STV.plot(kind='contourf')

# %%
# 4.3.5.3 :func`marginals <skgstat.SpaceTimeVariogram.marginals>`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A very important step for the estimation of spatio-temporal variogram models,
# is the estimation of marginal models. While the marginal models are
# implemented as :class:`Variogram <skgstat.Variogram>` instances and can be
# changed and plotted like any other :class:`Variogram <skgstat.Variogram>` instance,
# it can come very handy to plot the marginal models side-by-side.
# 
# This can be done with the :func`marginals <skgstat.SpaceTimeVariogram.marginals>` method.
backend('plotly')
fig = STV.marginals(show=False)
fig

# %%
# 
backend('matplotlib')
fig = STV.marginals()

# %%
# Additionally, the separated spatial and temporal models can be plotted into each sub-plot:
fig = STV.marginals(include_model=True)

# %%
#
backend('plotly')
fig = STV.marginals(include_model=True, show=False)
fig
