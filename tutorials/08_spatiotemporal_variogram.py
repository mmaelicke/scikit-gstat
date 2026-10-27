"""
8. Spatio-temporal Variograms
=============================
With version ``1.0.26``, SciKit-GStat implements the
:class:`SpatioTemporalVariogram <skgstat.SpatioTemporalVariogram>` class.
It replaces the :class:`SpaceTimeVariogram <skgstat.SpaceTimeVariogram>`, which
is deprecated now. The main difference is the input data: the old class needed
a complete grid of time series, one for each location. The new class takes one
row per observation, with the spatial coordinates followed by the time:

.. code-block:: text

    x, y, [z, ...], time

Thus, the observations can be located freely in space and time, which is the
common case for real sensor networks with missing values or different
sampling schedules. The time can be given as numbers or as datetime-like
objects.

The spatio-temporal data is derived from Fersch et al. (2020). From that data
publication, the wireless sensor network data is used. The originally published
15 minutes intervals soil temperature data at 20 cm depth was aggregated to
mean hourly values. Here, only 10 stations and every 12th hour is used, as the
number of observation pairs grows quadratically with the number of observations.
The data is published under Creative Commons BY 4.0 license. Please cite the
original publication if you use the data, and **not** SciKit-GStat.

**References**

Fersch, Benjamin, et al. "A dense network of cosmic-ray neutron sensors for soil moisture observation in a pre-alpine headwater catchment in Germany." Earth System Science Data Discussions 2020 (2020): 1-35.
"""
import skgstat as skg
import numpy as np
import pandas as pd
import json
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# %%
# 8.1 Load data
# -------------
# Load the TERENO soil temperature data from Fersch et al. (2020):
with open('./tereno_fendt/tereno.json', 'r') as js:
    data_obj = json.load(js)

coords = np.array(data_obj['coordinates'])
vals = np.array(data_obj['values'])
print(data_obj['description'])

# %%
# The data starts at May 1st, 2019. Use every 5th station and every 12th hour.
coords = coords[::5][:10]
vals = vals[::5][:10, ::12][:, :18]
times = pd.date_range('2019-05-01', periods=vals.shape[1], freq='12h')

# %%
# 8.2 From a grid
# ---------------
# The data is a grid of time series, as used by the old
# :class:`SpaceTimeVariogram <skgstat.SpaceTimeVariogram>`. The
# :func:`from_grid <skgstat.SpatioTemporalVariogram.from_grid>` method
# converts each grid cell into one observation.
STV = skg.SpatioTemporalVariogram.from_grid(coords, vals, times=times, x_lags=6, t_lags=8)
print('Observations: %d' % STV.values.size)
print('Pairs: %d' % STV.bin_count.sum())

# %%
# 8.3 Observations
# ----------------
# To illustrate irregular observations, we remove every 5th observation
# from the grid. The result is a table with one row per observation, which
# cannot be represented by the old grid interface anymore.
df = pd.DataFrame({
    'x': np.repeat(coords[:, 0], len(times)),
    'y': np.repeat(coords[:, 1], len(times)),
    'time': np.tile(times, len(coords)),
    'temperature': vals.flatten()
})
df = df.iloc[np.arange(len(df)) % 5 != 0].dropna()
print(df.head())

# %%
# The time is the last coordinate column. Datetime-like time is anchored
# at the oldest timestamp and converted into seconds. All lags are calculated
# in seconds, therefore ``maxlag_time`` is given in seconds as well. Only the
# plots convert the time axis into the unit of the data resolution.
STV = skg.SpatioTemporalVariogram(
    df[['x', 'y', 'time']],
    df['temperature'],
    x_lags=6,
    t_lags=8,
    maxlag_time=4 * 24 * 60 * 60
)
print('Time origin: %s' % STV.time_origin)
print('Time unit: %s' % STV.time_unit)

# %%
# 8.4 Experimental variogram
# --------------------------
# Each space and time lag class combination has its own number of point pairs.
# Check the bin count before interpreting the experimental variogram.
print(STV.bin_count)

# %%
# The ``matrix`` plot shows the experimental variogram for each lag class.
fig = STV.plot(kind='matrix', figsize=(8, 5))

# %%
# 8.5 Fitting
# -----------
# The product-sum model fits the marginal models for space and time first.
# Then, its three parameters are fitted to the experimental variogram.
STV.fit()
print(STV.cof)

# %%
# Once fitted, the contour plots show the model.
fig = STV.plot(kind='contourf', figsize=(8, 5), levels=15)

# %%
# The 3D plot shows the experimental variogram along with the model.
fig = STV.plot(kind='scatter', figsize=(8, 6))

# %%
# 8.6 Metric model
# ----------------
# The sum, product and product-sum models are separable: they are built from
# the two marginal variograms. These need pairs at the same location and at
# the same time, which irregular observations often do not have. The
# ``'metric'`` model rescales the time axis into a space distance and applies
# the ``marginal_model`` to the combined distance:
#
# .. math::
#     \gamma (h,t) = \gamma_m \left(\sqrt{h^2 + \left(\frac{r_x}{r_t}t\right)^2}\right)
#
# Here, :math:`r_x` and :math:`r_t` are the effective ranges along the space
# and time axis. The model is fitted to the experimental variogram directly.
# The 12 hour values follow a daily cycle, which the metric model cannot
# represent. Thus, we use the daily values at midnight for the full 15 days.
daily = np.array(data_obj['values'])[::5][:10, ::24]
days = pd.date_range('2019-05-01', periods=daily.shape[1], freq='D')

MTV = skg.SpatioTemporalVariogram.from_grid(
    coords,
    daily,
    times=days,
    model='metric',
    marginal_model='exponential'
)

# %%
# The :func:`parameters <skgstat.SpatioTemporalVariogram.parameters>` hold the
# fitted parameters by name. The time range is given in seconds.
params = MTV.parameters
print('Space range: %.1f' % params['space_range'])
print('Time range: %.1f days' % (params['time_range'] / 86400))
print('Sill: %.2f' % params['sill'])

# %%
# The space range is the largest space lag. Within the extent of the ten
# stations, the semi-variance does not reach a sill along the space axis.
fig = MTV.plot(kind='contourf', figsize=(8, 5), levels=15)

# %%
# 8.7 Kriging with GSTools
# ------------------------
# The metric model can be exported to a spatio-temporal GSTools
# :any:`CovModel <gstools.CovModel>`. The time is appended as last dimension,
# the length scales are the effective ranges of space and time. The separable
# models cannot be exported, as GSTools only supports metric space-time models.
model = MTV.to_gstools()
print(model)

# %%
# :func:`to_gs_krige <skgstat.SpatioTemporalVariogram.to_gs_krige>` returns a
# GSTools :any:`Krige <gstools.krige.Krige>` instance conditioned on all
# observations. The kriging targets have to use the same time unit as the
# variogram. For datetime input, these are the seconds since
# :func:`time_origin <skgstat.SpatioTemporalVariogram.time_origin>`.
# Here, we interpolate a time series every 3 hours at the first station.
krige = MTV.to_gs_krige()

targets = pd.date_range(days[0], days[-1], freq='3h')
t = (targets - MTV.time_origin).total_seconds()
x = np.full(t.size, coords[0, 0])
y = np.full(t.size, coords[0, 1])
field, variance = krige((x, y, t))

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.fill_between(targets, field - np.sqrt(variance), field + np.sqrt(variance), alpha=0.3, label='kriging std')
ax.plot(targets, field, '-b', label='kriging')
ax.plot(days, daily[0], 'or', label='observations')
ax.set_ylabel('soil temperature')
ax.legend()
fig.autofmt_xdate()

# %%
# The same instance interpolates a map at any time, here noon of May 8th,
# which lies between two observation days.
gx = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 50)
gy = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 50)
noon = (pd.Timestamp('2019-05-08 12:00') - MTV.time_origin).total_seconds()
field, variance = krige.structured((gx, gy, [noon]))

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
art = ax.imshow(field[:, :, 0].T, origin='lower', extent=(gx[0], gx[-1], gy[0], gy[-1]), cmap='RdYlBu_r')
ax.plot(coords[:, 0], coords[:, 1], '+k')
plt.colorbar(art, ax=ax)
