"""
3 - Variogram Models
====================

This tutorial will guide you through the theoretical variogram models available for the :class:`Variogram <skgstat.Variogram>` class. 

**In this tutorial you will learn:**

    * how to choose an appropiate model function
    * how to judge fitting quality
    * about sample size influence

"""
import skgstat as skg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
skg.plotting.backend('matplotlib')
# %%
# 3.1 Load data
# -------------
# For this example we will use the pancake dataset. You can use the
# :mod:``skgstat.data`` submodule to directly sample the dataset. This is the
# red-channel of an image of an actual pancake. The intersting thing about this pancake is,
# that it shows some clear spatial structures in its browning, but of different 
# shapes at different scales. This should be reflectable with different samples.
s = [30, 80, 300]
data1 = skg.data.pancake(N=s[0], seed=42, as_dataframe=True).get('sample')
data2 = skg.data.pancake(N=s[1], seed=42, as_dataframe=True).get('sample')
data3 = skg.data.pancake(N=s[2], seed=42, as_dataframe=True).get('sample')


# %%
# Plotting:
def plot_scatter(data, ax):
    art = ax.scatter(data.x, data.y, 50, c=data.v, cmap='plasma')
    plt.colorbar(art, ax=ax)

# run
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for data, ax in zip((data1, data2, data3), axes.flatten()):
    plot_scatter(data, ax)

# %%
# 3.2 Comparing theoretical models
# --------------------------------
# One of the features of :mod:`skgstat` is the fact that it is programmed object oriented.
# That means, we can just instantiate a :class:`Variogram <skgstat.Variogram>` object
# and start changing arguments unitl it models spatial dependency in our observations well.
V1 = skg.Variogram(data1[['x', 'y']].values, data1.v.values, maxlag='median', normalize=False)
V1.plot(show=False);

# %% 
# Plot the others as well
V2 = skg.Variogram(data2[['x', 'y']].values, data2.v.values, maxlag='median', normalize=False)
V3 = skg.Variogram(data3[['x', 'y']].values, data3.v.values, maxlag='median', normalize=False)

fig, _a = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
axes = _a.flatten()

x = np.linspace(0, V1.maxlag, 100)

# plot each variogram
for i, v in enumerate([V1, V2, V3]):
    axes[i].plot(v.bins, v.experimental, '.b')
    axes[i].plot(x, v.fitted_model(x), '-g')
    axes[i].set_title(f'N = {s[i]}')
    axes[i].set_xlabel('Lag (-)')
    if i == 0:
        axes[0].set_ylabel('semivariance (matheron)')
    axes[i].grid(which='major', axis='x')
plt.tight_layout()

# %%
# We can see how the experimental variogram changes dramatically with sample size.
# Depending on the sample size, we can also choose different number of lag classes.
# As the :class:`Variogram <skgstat.Variogram>`` is object oriented, we can simply
# update the binning function. First we set the number of lags directly, then we derive
# it from the distance matrix distribution. In the code below, we build the plot
# from scratch, demonstrating how you can access the empirical data and how it is updated, when new parameters are supplied.
fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey=True, sharex=True)

x = np.linspace(0, V1.maxlag, 100)
manual_lags = (6, 12, 18)
col_lab = ['10 lags', 'varying lags', 'Scott rule']

# plot each variogram
for i in range(3):
    for j, v in enumerate([V1, V2, V3]):
        # first row - use same settings
        if i == 0:
            v.bin_func = 'even'
            v.n_lags = 10
        # second row - use the manual lags
        if i == 1:
            v.n_lags = manual_lags[j]
        # last row - use scott
        if i == 2:
            v.bin_func = 'scott'
            axes[i][j].set_xlabel('Lag (-)')
        
        # plot
        axes[i][j].plot(v.bins, v.experimental, '.b')
        axes[i][j].plot(x, v.fitted_model(x), '-g')
        axes[i][j].grid(which='major', axis='x')
        
        # label first col
        if j == 0:
            axes[i][j].set_ylabel(col_lab[i])
        # title first row
        if i == 0:
            axes[i][j].set_title(f'N = {s[j]}')
plt.tight_layout()

# %%
# That actually demonstrates how the selection of the experimental variogram can 
# have huge influence on the base data for fitting. Now consider the center column.
# In each of the plots, the selection of model is not deterministic.
# You can argue for at least two different models here, that might actually be supported by the empirical data.
# The :class:`Variogram <skgstat.Variogram>` class has several goodness of fit
# measures to help you on assessing the fit. This does not replace a careful
# visual inspection of the models, but can assist you in making an decision.
# Remember that the Kriging will be influenced by the quality of the spatial model,
# especially on short distances.
# We can quickly cycle all available models for the sample size of 80 to see
# if spherical fits best. The histogram plot can be turned off.

# we use the settings from before - scott rule
V2.bin_func = 'scott'
fig, _a = plt.subplots(2,3, figsize=(12, 6), sharex=True, sharey=True)
axes = _a.flatten()
for i, model in enumerate(('spherical', 'exponential', 'gaussian', 'matern', 'stable', 'cubic')):
    V2.model = model
    V2.plot(axes=axes[i], hist=False, show=False)
    axes[i].set_title('Model: %s; RMSE: %.2f' % (model, V2.rmse))
    axes[i].set_ylim(0, 2000)

# %%
# This is quite important. We find all 6 models to describe the experimental
# variogram more or less equally well in terms of RMSE. Think of the
# implications: We basically can use any model we like. 
# This is a problem as i.e. the gaussian and the spherical model describe
# fundamentally different spatial properties. Thus, our model selection
# should be driven by interpretation of the variogram, and not the difference
# in RMSE of only 0.4%, which might very likely not be significant at all.
# 
# 
# But what does this difference look like, when it comes to interpolation?

def interpolate(V, ax):
    xx, yy = np.mgrid[0:499:100j, 0:499:100j]
    ok = skg.OrdinaryKriging(V, min_points=5, max_points=15, mode='exact')
    field = ok.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
    art = ax.matshow(field, origin='lower', cmap='plasma', vmin=V.values.min(), vmax=V.values.max())
    ax.set_title('%s model' % V.model.__name__)
    plt.colorbar(art, ax=ax)
    return field

# %%

fields = []
fig, _a = plt.subplots(2,3, figsize=(12, 10), sharex=True, sharey=True)
axes = _a.flatten()
for i, model in enumerate(('spherical', 'exponential', 'gaussian', 'matern', 'stable', 'cubic')):
    V2.model = model
    fields.append(interpolate(V2, axes[i]))


# %%
# Get some basic statistics about the fields

pd.DataFrame({'spherical': fields[0].flatten(), 'exponential': fields[1].flatten(), 'gaussian': fields[2].flatten(),
              'matern': fields[3].flatten(), 'stable': fields[4].flatten(), 'cubic': fields[5].flatten()}).describe()

# %%
# This should illustrate, how important the selection of model is, even if no observation uncertainties are propagated into the analysis.
# 
#   1. Gaussian model is far off, producing estimations far outside the observed value ranges
#   2. All other models seem produce quite comparable mean values
#   3. BUT: the standard deviation is quite different
#   4. The median of the field can vary by more than 3 units, even if we took the Gaussian model out
# 
# You have to remind that we had quite some observations. The selection of model becomes even more arbitrary with smaller samples and more importantly: We have to consider more than one equally probable parameterization of each model when the experimental is more spreaded.

# Finally, we can calculate the difference between the kriging fields to inspect the spread of estimations spatially:
#
field_min = np.nanmin(np.stack(fields, axis=2), axis=2)
field_max = np.nanmax(np.stack(fields, axis=2), axis=2)

fig, ax = plt.subplots(1, 1, figsize=(7,7))
m = ax.matshow(field_max - field_min, origin='lower', cmap='Reds')
plt.colorbar(m)

# %%
# The colorbar is spanning the entire value range. Thus, given the minor differences in the fitting of the models, we would have to reject just any estimation based on an automatic fit, which is considering some uncertainties in the selection of parameters, because the RMSE values were too close.
# 
# To use the result from above, we need to justfy the selection of model first and manually fit the model based on expert knowledge.

# %%
# 3.3 Using other sample sizes
# ----------------------------
# Let's have a look at the sparse sample again
V1.plot(show=False);

# %%
# This is a nugget-effect variogram. Thus we have to reject any geostatistical 
# analysis based on this sample. It just does not expose any spatial pattern that 
# can be exploited.
# 
# What about the denser sample. Increasing the sample size should reject some 
# of the models. Remind, that we are sampling at more short distances and thus,
# the variogram will be governed by the short ranged patterns of the field, while
# the other samples are more dependent on the medium and large range patterns, as
# there were less short separating distances sampled.

# we use the settings from before - scott rule
V3.bin_func = 'scott'
fig, _a = plt.subplots(2,3, figsize=(12, 6), sharex=True, sharey=True)
axes = _a.flatten()
for i, model in enumerate(('spherical', 'exponential', 'gaussian', 'matern', 'stable', 'cubic')):
    V3.model = model
    V3.plot(axes=axes[i], hist=False, show=False)
    axes[i].set_title('Model: %s; RMSE: %.2f' % (model, V3.rmse))
    axes[i].set_ylim(0, 2000)

# %%
# We can now clearly reject the cubic, gaussian and exponential model. 
# I personally would also reject the spherical model we used in the fist place,
# as it is systematically underestimating the semi-variance on short distances. 
d_fields = []
fig, _a = plt.subplots(2,3, figsize=(18, 12), sharex=True, sharey=True)
axes = _a.flatten()
for i, model in enumerate(('spherical', 'exponential', 'gaussian', 'matern', 'stable', 'cubic')):
    V3.model = model
    d_fields.append(interpolate(V3, axes[i]))


# %%
# Again some statistics

pd.DataFrame({'spherical': d_fields[0].flatten(), 'exponential': d_fields[1].flatten(), 'gaussian': d_fields[2].flatten(),
              'matern': d_fields[3].flatten(), 'stable': d_fields[4].flatten(), 'cubic': d_fields[5].flatten()}).describe()

# %%
# Finally, if we only concentrate on the not-rejected models: matern and stable,
# we can see hardly any difference in the field. Additionally, except for extrema,
# the statistical properties of the two fields are largely the same.
