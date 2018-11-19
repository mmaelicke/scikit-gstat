===========
Variography
===========

The variogram
=============

General
-------

We start by constructing a random field and sample it. Without knowing about
random field generators, an easy way to go is to stick two trigonometric
functions together and add some noise. There should be clear spatial
correlation apparent.

.. ipython:: python

    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

This field could look like

.. ipython:: python

    # apply the function to a meshgrid and add noise
    xx, yy = np.mgrid[0:0.5 * np.pi:500j, 0:0.8 * np.pi:500j]
    np.random.seed(42)

    # generate a regular field
    _field = np.sin(xx)**2 + np.cos(yy)**2 + 10

    # add noise
    np.random.seed(42)
    z = _field + np.random.normal(0, 0.15, (500,  500))

    @savefig tf.png width=8in
    plt.imshow(z, cmap='RdYlBu_r')

Using scikit-gstat
~~~~~~~~~~~~~~~~~~

It's now easy and straightforward to calculate a variogram using
``scikit-gstat``. We need to sample the field and pass the coordinates and
value to the :class:`Variogram Class <skgstat.Variogram>`.

.. ipython:: python
    :okwarning:

    from skgstat import Variogram

    # random coordinates
    np.random.seed(42)
    coords = np.random.randint(0, 500, (300, 2))
    values = np.fromiter((z[c[0], c[1]] for c in coords), dtype=float)

    V = Variogram(coords, values)
    @savefig var1.png width=8in
    V.plot()

From my personal point of view, there are three main issues with this approach:

* If one is not an geostatistics expert, one has no idea what he actually did
  and can see in the presented figure.
* The figure includes an spatial model, one has no idea if this model is
  suitable and fits the observations (wherever they are in the figure)
  sufficiently.
* Refer to the :func:`__init__ <skgstat.Variogram.__init__>` method of the
  Variogram class. There are 10+ arguments that can be set optionally. The
  default values will most likely not fit your data and requirements.

Therefore one will have to understand how the
:class:`Variogram Class <skgstat.Variogram>` works along with some basic
knowledge about variography in oder to be able to properly use ``scikit-gstat``.

However, what we can discuss from the figure, is what a variogram actually is
At its core it relates a dependent variable to an independent variable and,
in a second step, tries to describe this relationship with a statistical
model. This model on its own describes some of the spatial properties of the
random field and can further be utilized in an interpolation to select nearby
points and weight them based on their statistical properties.

The variogram relates the separating distance between two observation points
to a measure of variability of values at that given distance. Our expectation
is that variance is increasing with distance, what can basically be seen in
the presented figure.

Distance
--------

Consider the variogram figure from above, with which an *independent* and
*dependent* variable was introduced. In statistics it is common to use
*dependent* variable as an alias for *target variable*, because its value is
dependent on the state of the independent variable. In the case of a
variogram, this is the metric of variance on the y-axis. The independent
variable is a measure of (usually) Euclidean distance.

Consider observations taken in the environment, it is fairly unlikely to find
two pairs of observations where the separating distance between the
coordinates match exactly the same value. Therefore it is useful to group all
point pairs at the same distance *lag* together into one group, or *bin*.
Beside practicability, there is also another reason, why one would want to
group point pairs at similar separating distances together into one bin.
This becomes obvious, when one plots the difference in value over the
distance for all point pair combinations that can be formed for a given sample.
The :class:`Variogram Class <skgstat.Variogram>` has a function for that:
:func:`distance_difference_plot <skgstat.Variogram.distance_difference_plot>`:

.. ipython:: python
    :okwarning:

    @savefig dist_diff_plot.png width=8in
    V.distance_difference_plot()

While it is possible to see the increasing variability with increasing
distance here quite nicely, it is not possible to guess meaningful moments
for the distributions within the bins. Last but not least, to derive a simple
model as presented in the variogram figure above by the green line, we have
to be able to compress all values at a given distance lag to one estimation
of variance. This would not be possible from the the figure above.

.. note::

    There are also procedures that can fit a model directly based on unbinned
    data. As none of these methods is implemented into ``scikit-gstat``, they
    will not be discussed here. If you need them, you are more than welcome
    to implement them. Else you'll have to wait until I did that.

Binning the separating distances into distance lags is therefore a crucial and
most important task in a variogram analysis. The final binning shall
discretizise the distance lag at a meaningful resolution at the scale of
interest while still holding enough members in the bin to make valid
estimations. Often this is a trade-off relationship and one has to find a
suitable compromise.

Before diving into binning, we have to understand how the
:class:`Variogram Class <skgstat.Variogram>` handles distance data. The
distance calculation can be controlled by the ``dist_func`` argument, which
takes either a string or a function. The default value is `'euclidean'`.
This value is directly passed down to the
:func:`pdist <scipy.spatial.distance.pdist>` as the `metric` argument.

``scikit-gstat`` has two different methods for binning.

.. note::

    This is not finished. Will continue whenever I can find some time.

Observation differences
-----------------------

Experimental variograms
-----------------------

When direction matters
======================

What is 'direction'?
--------------------


Space-time variography
======================
