"""
6 - GSTools
===========
With version ``0.5`` ``scikit-gstat`` offers an interface to the awesome `gstools <https://github.com/GeoStat-Framework/GSTools>`_
library. This way, you can use a :class:`Variogram <skgstat.Variogram>` estimated with ``scikit-gstat`` 
in `gstools <https://github.com/GeoStat-Framework/GSTools>`_  to perform random field generation, kriging and much, much more.

For a :class:`Variogram <skgstat.Variogram>` instance, there are three possibilities to export into `gstools <https://github.com/GeoStat-Framework/GSTools>`_ : 

    1. :func:`Variogram.get_empirical(bin_center=True) <skgstat.Variogram.get_empirical>` returns a pair of distance lag bins and experimental semi-variance values, like `gstools.variogram.vario_estimate <https://geostat-framework.readthedocs.io/projects/gstools/en/latest/generated/gstools.variogram.vario_estimate.html>`_. 
    2. :func:`Variogram.to_gstools <skgstat.Variogram.to_gstools>` returns a parameterized :any:`CovModel <gstools.covmodel.CovModel>` derived from the Variogram.
    3. :func:`Variogram.to_gs_krige <skgstat.Variogram.to_gs_krige>` returns a :any:`GSTools Krige <gstools.krige.Krige>` instance based on the variogram


6.1 ``get_empirical``
---------------------

6.1.1 Reproducing the gstools example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can reproduce the `Getting Started example for variogram estimation from GSTools docs <https://geostat-framework.readthedocs.io/projects/gstools/en/latest/index.html#id3>`_ 
with ``scikit-gstat``, and replace the calculation of the empirical variogram with :class:`skg.Variogram <sggstat.Variogram>`. 

Note: This does only make sense if you want to use a distance metric, binning procedure or semi-variance estimator, that is not included in `gstools` or are bound to `scikit-gstat` for any other reason. :class:`Variogram <skgstat.Variogram>` will _always_ perform a full model fitting cycle on instantiation, which could lead to some substantial overhead here.
This behavior might change in a future version of `scikit-gstat`.

"""

# %% Import all modules

# import
import skgstat as skg
import gstools as gs
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
skg.plotting.backend('matplotlib')

# %%
# use the example from gstools
# generate a synthetic field with an exponential model

x = np.random.RandomState(19970221).rand(1000) * 100.
y = np.random.RandomState(20011012).rand(1000) * 100.
model = gs.Exponential(dim=2, var=2, len_scale=8)
srf = gs.SRF(model, mean=0, seed=19970221)
field = srf((x, y))

# %%
# combine x and y for use in skgstat
coords = np.column_stack((x, y))

# %%
# In the example, :any:`gstools.variogram.vario_estimate` is used to estimate the empirical variogram:
#
# .. code-block:: python
#
#   # estimate the variogram of the field
#   bin_center, gamma = gs.vario_estimate((x, y), field)
#
#
# Here, we can use :class:`skg.Variogram <skgstat.Variogram>`. 
# From the shown arguments, :func:`estimator <skgstat.Variogram.estimator>` and
# :func:`bin_func <skgstat.Variogram.bin_func>` are using the default values:

V = skg.Variogram(coords, field, n_lags=21, estimator='matheron', maxlag=45, bin_func='even')
bin_center, gamma = V.get_empirical(bin_center=True)


# %%
# And finally, the exact same code from the GSTools docs can be called:
# fit the variogram with a stable model. (no nugget fitted)

fit_model = gs.Stable(dim=2)
fit_model.fit_variogram(bin_center, gamma, nugget=False)

# %%
# Output the model
ax = fit_model.plot(x_max=max(bin_center))
ax.scatter(bin_center, gamma)
print(fit_model)

# %%
#
# 6.1.2 ``bin_center=False``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# It is important to understand, that ``gstools`` and ``skgstat`` are handling lag bins different.
# While ``skgstat`` uses the upper limit, ``gstools`` assumes the bin center.
# This can have implications, if a model is fitted. C
# onsider the example below, in which only the ``bin_center`` setting is different.

bin_edges, _ = V.get_empirical(bin_center=False)

# fit the variogram with a stable model. (no nugget fitted)
edge_model = gs.Stable(dim=2)
_ = edge_model.fit_variogram(bin_edges, gamma, nugget=False)


# %%
# Make a nice plot


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# plot first
fit_model.plot(ax=axes[1], label='center=True')
# plot second
edge_model.plot(ax=axes[1], label='center=False')

# bins
axes[0].scatter(bin_center, gamma, label='center=True')
axes[0].scatter(bin_edges, gamma, label='center=False')

axes[0].set_title('Empirical Variogram')
axes[1].set_title('Variogram Model')
axes[0].legend(loc='lower right')
print(fit_model)
print(edge_model)

# %%
# Notice the considerable gap between the two model functions. This can already lead to seroius differences, i.e. in Kriging.

# %%
# 6.1.3 Using other arguments
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now, with the example `from the GSTools docs <https://geostat-framework.readthedocs.io/projects/gstools/en/latest/index.html#id3>`_ working,
# we can start chaning the arguments to create quite different empirical variograms.
#
# **Note**: This should just illustrate the available possibilities, the result is by no means producing a better
# estimate of the initially created Gaussian random field.
#
# In this example different things will be changed:
#
# - use only 15 lag classes, but distribute the point pairs equally. Note the differing widths of the classes. (``bin_func='uniform'``)
# - The :func:`Dowd <skgstat.estimators.dowd>` estimator is used. (``estimator='dowd'``)
# - The Taxicab metric (https://en.wikipedia.org/wiki/Taxicab_geometry) (aka. Manhattan metric or cityblock metric) is used over
# Euklidean for no obvious reason. (``dist_func='cityblock'``)

V = skg.Variogram(coords, field, n_lags=15, estimator='dowd', maxlag=45, bin_func='uniform', dist_func='cityblock')
bin_center, gamma = V.get_empirical(bin_center=True)


# %%
# fit the variogram with a stable model. (no nugget fitted)
fit_model = gs.Stable(dim=2)
fit_model.fit_variogram(bin_center, gamma, nugget=True)

# output
ax = fit_model.plot(x_max=max(bin_center))
ax.scatter(bin_center, gamma)
print(fit_model)

# %%
# If you fit the `gs.Stable` with a nugget, it fits quite well. But keep in mind that this does not necessarily describe the original field very well and was just fitted for demonstration.

# %%
# 6.2 ``to_gstools``
# ~~~~~~~~~~~~~~~~~~
#
# The second possible interface to ``gstools`` is the :func:`Variogram.to_gstools <skgstat.Variogram.to_gstools>` function.
# This will return one of the classes `listed in the gstools documentation <https://geostat-framework.readthedocs.io/projects/gstools/en/latest/package.html#covariance-models>`_.
# The variogram parameters are extracted and passed to gstools. You should be able to use it, just like any other :any:`CovModel <gstools.covmodel.CovModel>`.
#
# However, there are a few things to consider:
#
# - ``skgstat`` can only export isotropic models.
# - The ``'harmonize'`` cannot be exported
#
# 6.2.1 exporting :class:`Variogram <skgstat.Variogram>`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In this example, the same Variogram from above is estimated, but we use the :func:`exponential <skgstat.models.exponential>` model. 
# An exponential covariance function was used in the first place to create the field that was sampled.

skg.plotting.backend('plotly')
V = skg.Variogram(coords, field, n_lags=21, estimator='matheron', model='exponential', maxlag=45, bin_func='even')
V.plot()

# %%
# Now export the model to ``gstools``:

exp_model = V.to_gstools()
print(exp_model)

# get the empirical for the plot as well
bins, gamma = V.get_empirical(bin_center=True)

ax = exp_model.plot(x_max=45)
ax.scatter(bins, gamma)

# %%
# **Note:** It is important to understand, that ``skgstat`` and ``gstools`` handle coordinates slightly different.
# If you export the :class:`Variogram <skgstat.Variogram>` to a :any:`CovModel <gstools.covmodel.CovModel>`
# and you want to use the :class:`Variogram.coordinates <skgstat.Variogram.coordinates>`, you **must** transpose them.
#
# .. code-block:: python
# 
#   # variogram is a skgstat.Variogram instance
#   model = variogram.to_gstools()
#   cond_pos = variogram.coordinates.T
#
#   # use i.e. in Kriging
#   krige = gs.krige.Ordinary(model, cond_pos, variogram.values)
#
# 6.2.2 Spatial Random Field Generation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# With a :any:`CovModel <gstools.covmodel.CovModel>`, we can use any of the great tools implemented in ``gstools``.
# First, let's create another random field with the exponential model that we exported in the last section:

x = y = range(100)
new_field = gs.SRF(exp_model, seed=13062018)
new_field.structured([x, y])
new_field.plot()

# %%
# Keep in mind, that we did not call a Kriging procedure, but created **another** field.
# Of course, we can do the same thing with the more customized model, created in 6.1.3:
malformed = gs.SRF(fit_model, seed=24092013)
malformed.structured([x, y])
malformed.plot()

# %%
# Notice how the spatial properties as well as the value range has changed. 
# That's why it is important to estimate :class:`Variogram <skgstat.Variogram>` or :any:`CovModel <gstools.covmodel.CovModel>` 
# carefully and not let the GIS do that for you somewhere hidden in the dark.

# %%
# 6.3 ``to_gs_krige``
# ~~~~~~~~~~~~~~~~~~~
#
# Finally, after carefully esitmating and fitting a variogram using SciKit-GStat, 
# you can also export it directly into a :any:`GSTools Krige <gstools.krige.Krige>` instance. 
# We use the variogram as in the other sections:

# export
krige = V.to_gs_krige(unbiased=True)  # will result in ordinary kriging
print(krige)

# create a regular grid
x = y = range(100)

# interpolate
result, sigma = krige.structured((x, y))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# plot
axes[0].imshow(result, origin='lower')
axes[1].imshow(sigma, origin='lower', cmap='RdYlGn_r')

# label
axes[0].set_title('Kriging')
axes[1].set_title('Error Variance')

plt.tight_layout()
