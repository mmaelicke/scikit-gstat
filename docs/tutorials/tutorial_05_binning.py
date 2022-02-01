"""
5 - Lag classes
===============
This tutorial focuses the estimation of lag classes. It is one of the most
important, maybe **the** most important step for estimating variograms.
Usually, lag class generation, or binning, is not really focused in
geostatistical literature. The main reason is, that usually, the same method is
used. A user-set amount of equidistant lag classes is formed with ``0`` as lower
bound and ``maxlag`` as upper bound. Maxlag is often set to the median or 60%
percentile of all pairwise separating distances. 

In SciKit-GStat this is also the default behavior, but only one of dozen of
different implemented methods. Thus, we want to shed some light onto the other
methods here. SciKit-GStat implements methods of two different kinds. The
first kind are the methods, that take a fixed `N`, the number of lag classes,
accessible through the :func:`Variogram.n_lags <skgstat.Variogram.n_lags>`
property. These methods are ``['even', 'uniform', 'kmeans', 'ward']``.
The other kind is often used in histogram estimation and will apply a (simple)
rule to figure out a suitable `N` themself. Using one of these methods will
overwrite the :func:`Variogram.n_lags <skgstat.Variogram.n_lags>` property.
THese methods are: ``['sturges', 'scott', 'fd', 'sqrt', 'doane']``.

"""
import skgstat as skg
import numpy as np
import pandas as pd
from imageio import imread
import plotly.graph_objects as go
from plotly.subplots import make_subplots

skg.plotting.backend('plotly')

# %%
# 5.1 Sample data
# ---------------
# Loads a data sample and draws `n_samples` from the field.
# For sampling the field, random samples from a gamma distribution with a fairly
# high scale are drawn, to ensure there are some outliers in the samle. The
# values are then re-scaled to the shape of the random field and the values
# are extracted from it.
# You can use either of the next two cell to work either on the pancake or
# the Meuse dataset.
N = 80
pan = skg.data.pancake_field().get('sample')
coords, vals = skg.data.pancake(N=80, seed=1312).get('sample')
fig = make_subplots(1,2,shared_xaxes=True, shared_yaxes=True)
fig.add_trace(
    go.Scatter(x=coords[:,0], y=coords[:,1], mode='markers', marker=dict(color=vals,cmin=0, cmax=255), name='samples'),
    row=1, col=1
)
fig.add_trace(go.Heatmap(z=pan, name='field'), row=1, col=2)
fig.update_layout(width=900, height=450, template='plotly_white')
fig

# %%
# .. note:: 
#   You need to comment the next cell to use the pancake dataset. This cell will
#   will overwrite the ``coords`` and ``vals`` array create in the last cell.
coords, vals = skg.data.meuse().get('sample')
vals = vals.flatten()
fig = go.Figure(go.Scatter(x=coords[:,0], y=coords[:,1], mode='markers', marker=dict(color=vals), name='samples'))
fig.update_layout(width=450, height=450, template='plotly_white')
fig

# %%
# 5.2 Lag class binning - fixed ``N``
# -----------------------------------
# Apply different lag class binning methods and visualize their histograms.
# In this section, the distance matrix between all point pair combinations
# ``(NxN)`` is binned using each method. The plots visualize the histrogram of
# the distance matrix of the variogram, **not** the variogram lag classes
# themselves.
N = 15

# use a nugget
V = skg.Variogram(coords, vals, n_lags=N, use_nugget=True)

# %%
# 5.2.1 default :func:`'even' <skgstat.binning.even_width_lags>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The default binning method will find ``N`` equidistant bins. This is the
# default behavior and used in almost all geostatistical publications.
# It should not be used without a ``maxlag`` (like done in the plot below).

# apply binning
bins, _ = skg.binning.even_width_lags(V.distance, N, None)

# get the histogram
count, _ = np.histogram(V.distance, bins=bins)

fig = go.Figure(
    go.Bar(x=bins, y=count),
    layout=dict(template='plotly_white', title=r"$\texttt{'even'}~~binning$")
)
fig

# %%
# 5.2.2 :func:`'uniform' <skgstat.binning.uniform_count_lags>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The histogram of the :func:`'uniform' <skgstat.binning.uniform_count_lags>`
# method will adjust the lag class widths to have the same sample size for each
# lag class. This can be used, when there must not be any empty lag classes on
# small data samples, or comparable sample sizes are desireable for the
# semi-variance estimator. 

# apply binning
bins, _ = skg.binning.uniform_count_lags(V.distance, N, None)

# get the histogram
count, _ = np.histogram(V.distance, bins=bins)

fig = go.Figure(
    go.Bar(x=bins, y=count),
    layout=dict(template='plotly_white', title=r"$\texttt{'uniform'}~~binning$")
)

fig

# %%
# 5.2.3 :func:`'kmeans' <skgstat.binning.kmeans>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The distance matrix is clustered by a K-Means algorithm.
# The centroids are used as lag class centers. Each lag class is then formed
# by taking half the distance to each sorted neighboring centroid as a bound. 
# This will most likely result in non-equidistant lag classes.
# 
# One important note about K-Means clustering is, that it is not a
# deterministic method, as the starting points for clustering are taken randomly.
# Thus, the decision was made to seed the random start values. Therefore, the
# K-Means implementation in SciKit-GStat is deterministic and will always
# return the same lag classes for the same distance matrix.

# apply binning
bins, _ = skg.binning.kmeans(V.distance, N, None)

# get the histogram
count, _ = np.histogram(V.distance, bins=bins)

fig = go.Figure(
    go.Bar(x=bins, y=count),
    layout=dict(template='plotly_white', title=r"$\texttt{'K-Means'}~~binning$")
)

fig

# %%
# 5.2.4 :func:`'ward' <skgstat.binning.ward>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The other clustering algorithm is a hierarchical clustering algorithm.
# This algorithm groups values together based on their similarity, which is
# expressed by Ward's criterion. 
# Agglomerative algorithms work iteratively and deterministic, as at first
# iteration each value forms a cluster on its own. Each cluster is then merged
# with the most similar other cluster, one at a time, until all clusters are
# merged, or the clustering is interrupted. 
# Here, the clustering is interrupted as soon as the specified number of lag
# classes is reached. The lag classes are then formed similar to the K-Means
# method, either by taking the cluster mean or median as center.
# 
# Ward's criterion defines the one other cluster as the closest, that results
# in the smallest intra-cluster variance for the merged clusters. 
# The main downside is the processing speed. You will see a significant
# difference for ``'ward'`` and should not use it on medium and large datasets.

# apply binning
bins, _ = skg.binning.ward(V.distance, N, None)

# get the histogram
count, _ = np.histogram(V.distance, bins=bins)

fig = go.Figure(
    go.Bar(x=bins, y=count),
    layout=dict(template='plotly_white', title=r"$\texttt{'ward'}~~binning$")
)

fig

# %%
#  5.3 Lag class binning - adjustable ``N``
# -----------------------------------------
# 
# 5.3.1 :func:`'sturges' <skgstat.binning.auto_derived_lags>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Sturge's rule is well known and pretty straightforward. It's the defaul
#  method for histograms in R. The number of equidistant lag classes is defined like:
# 
# .. math::
# 
#   n =log_2 (x + 1)
# 
# Sturge's rule works good for small, normal distributed datasets.

# apply binning
bins, n = skg.binning.auto_derived_lags(V.distance, 'sturges', None)

# get the histogram
count, _ = np.histogram(V.distance, bins=bins)

fig = go.Figure(
    go.Bar(x=bins, y=count),
    layout=dict(template='plotly_white', title=r"$\texttt{'sturges'}~~binning~~%d~classes$" % n)
)

fig

# %%
# 5.3.2 :func:`'scott' <skgstat.binning.auto_derived_lags>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Scott's rule is another quite popular approach to estimate histograms.
# The rule is defined like:
# 
# .. math::
# 
#   h = \sigma \frac{24 * \sqrt{\pi}}{x}^{\frac{1}{3}}
# 
# Other than Sturge's rule, it will estimate the lag class width from the
# sample size standard deviation. Thus, it is also quite sensitive to outliers. 

# apply binning
bins, n = skg.binning.auto_derived_lags(V.distance, 'scott', None)

# get the histogram
count, _ = np.histogram(V.distance, bins=bins)

fig = go.Figure(
    go.Bar(x=bins, y=count),
    layout=dict(template='plotly_white', title=r"$\texttt{'scott'}~~binning~~%d~classes$" % n)
)

fig

# %%
# 5.3.3 :func:`'sqrt' <skgstat.binning.auto_derived_lags>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The only advantage of this method is its speed. The number of lag classes 
# is simply defined like:
# 
# .. math::
#   
#   n = \sqrt{x} $$
# 
# Thus, it's usually not really a good choice, unless you have a lot of samples.

# apply binning
bins, n = skg.binning.auto_derived_lags(V.distance, 'sqrt', None)

# get the histogram
count, _ = np.histogram(V.distance, bins=bins)

fig = go.Figure(
    go.Bar(x=bins, y=count),
    layout=dict(template='plotly_white', title=r"$\texttt{'sqrt'}~~binning~~%d~classes$" % n)
)

fig

# %%
# 5.3.4 :func:`'fd' <skgstat.binning.auto_derived_lags>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# The Freedman-Diaconis estimator can be used to derive the number of lag
# classes again from an optimal lag class width like: 
#
# .. math:: 
# 
#   h = 2\frac{IQR}{x^{1/3}}
# 
# As it is based on the interquartile range (IQR), it is very robust to outlier.
# That makes it a suitable method to estimate lag classes on non-normal distance
# matrices. On the other side it usually over-estimates the $n$ for small
# datasets. Thus it should only be used on medium to small datasets.

# apply binning
bins, n = skg.binning.auto_derived_lags(V.distance, 'fd', None)

# get the histogram
count, _ = np.histogram(V.distance, bins=bins)

fig = go.Figure(
    go.Bar(x=bins, y=count),
    layout=dict(template='plotly_white', title=r"$\texttt{'fd'}~~binning~~%d~classes$" % n)
)

fig

# %%
# 5.3.5 :func:`'doane' <skgstat.binning.auto_derived_lags>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Doane's rule is an extension to Sturge's rule that takes the skewness of the
# distance matrix into account. It was found to be a very reasonable choice on
# most datasets where the other estimators didn't yield good results.
# 
# It is defined like:
# 
# .. math::
#     \begin{split}
#         n = 1 + \log_{2}(s) + \log_2\left(1 + \frac{|g|}{k}\right) \\
#             g = E\left[\left(\frac{x - \mu_g}{\sigma}\right)^3\right]\\
#             k = \sqrt{\frac{6(s - 2)}{(s + 1)(s + 3)}}
#     \end{split}

# apply binning
bins, n = skg.binning.auto_derived_lags(V.distance, 'doane', None)

# get the histogram
count, _ = np.histogram(V.distance, bins=bins)

fig = go.Figure(
    go.Bar(x=bins, y=count),
    layout=dict(template='plotly_white', title=r"$\texttt{'doane'}~~binning~~%d~classes$" % n)
)

fig


# 5.4 Variograms
# --------------
# The following section will give an overview on the influence of the chosen
# binning method on the resulting variogram. All parameters will be the same for
# all variograms, so any change is due to the lag class binning. The variogram
# will use a maximum lag of ``200`` to get rid of the very thin last bins at
# large distances.
# 
# The ``maxlag`` is very close to the effective range of the variogram, thus you
# can only see differences in sill. But the variogram fitting is not at the
# focus of this tutorial. You can also change the parameter and fit a more
# suitable spatial model

# use a exponential model
V.set_model('spherical')

# set the maxlag
V.maxlag = 'median'

# %%
# 5.4.1 :func:`'even' <skgstat.binning.even_width_lags>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# set the new binning method
V.bin_func = 'even'

# plot
fig = V.plot(show=False)
print(f'"{V._bin_func_name}" - range: {np.round(V.cof[0], 1)} sill: {np.round(V.cof[1], 1)}')
fig.update_layout(template='plotly_white')
fig

# %%
# 5.4.2 :func:`'uniform' <skgstat.binning.uniform_count_lags>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# set the new binning method
V.bin_func = 'uniform'

# plot
fig = V.plot(show=False)
print(f'"{V._bin_func_name}" - range: {np.round(V.cof[0], 1)} sill: {np.round(V.cof[1], 1)}')
fig.update_layout(template='plotly_white')
fig

# %%
# 5.4.3 :func:`'kmeans' <skgstat.binning.kmeans>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 

# set the new binning method
V.bin_func = 'kmeans'

# plot
fig = V.plot(show=False)
print(f'"{V._bin_func_name}" - range: {np.round(V.cof[0], 1)} sill: {np.round(V.cof[1], 1)}')
fig.update_layout(template='plotly_white')
fig

# %%
# 5.4.4 :func:`'ward' <skgstat.binning.ward>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# set the new binning method
V.bin_func = 'ward'

# plot
fig = V.plot(show=False)
print(f'"{V._bin_func_name}" - range: {np.round(V.cof[0], 1)} sill: {np.round(V.cof[1], 1)}')
fig.update_layout(template='plotly_white')
fig

# %%
# 5.4.5 :func:`'sturges' <skgstat.binning.auto_derived_lags>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# set the new binning method
V.bin_func = 'sturges'

# plot
fig = V.plot(show=False)
print(f'"{V._bin_func_name}" adjusted {V.n_lags} lag classes - range: {np.round(V.cof[0], 1)} sill: {np.round(V.cof[1], 1)}')
fig.update_layout(template='plotly_white')
fig


# %%
# 5.4.6 :func:`'scott' <skgstat.binning.auto_derived_lags>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# set the new binning method
V.bin_func = 'scott'

# plot
fig = V.plot(show=False)
print(f'"{V._bin_func_name}" adjusted {V.n_lags} lag classes - range: {np.round(V.cof[0], 1)} sill: {np.round(V.cof[1], 1)}')
fig.update_layout(template='plotly_white')
fig



# %%
# 5.4.7 :func:`'fd' <skgstat.binning.auto_derived_lags>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

# set the new binning method
V.bin_func = 'fd'

# plot
fig = V.plot(show=False)
print(f'"{V._bin_func_name}" adjusted {V.n_lags} lag classes - range: {np.round(V.cof[0], 1)} sill: {np.round(V.cof[1], 1)}')
fig.update_layout(template='plotly_white')
fig


# %%
# 5.4.8 :func:`'sqrt' <skgstat.binning.auto_derived_lags>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# set the new binning method
V.bin_func = 'sqrt'

# plot
fig = V.plot(show=False)
print(f'"{V._bin_func_name}" adjusted {V.n_lags} lag classes - range: {np.round(V.cof[0], 1)} sill: {np.round(V.cof[1], 1)}')
fig.update_layout(template='plotly_white')
fig


# %%
# 5.4.9 :func:`'doane' <skgstat.binning.auto_derived_lags>` lag classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# In[23]:


# set the new binning method
V.bin_func = 'doane'

# plot
fig = V.plot(show=False)
print(f'"{V._bin_func_name}" adjusted {V.n_lags} lag classes - range: {np.round(V.cof[0], 1)} sill: {np.round(V.cof[1], 1)}')
fig.update_layout(template='plotly_white')
fig

