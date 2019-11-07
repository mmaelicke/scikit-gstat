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
most important task in variogram analysis. The final binning must
discretizise the distance lag at a meaningful resolution at the scale of
interest while still holding enough members in the bin to make valid
estimations. Often this is a trade-off relationship and one has to find a
suitable compromise.

Before diving into binning, we have to understand how the
:class:`Variogram Class <skgstat.Variogram>` handles distance data. The
distance calculation can be controlled by the 
:func:`dist_func <skgstat.Variogram.dist_func>` argument, which
takes either a string or a function. The default value is `'euclidean'`.
This value is directly passed down to the
:func:`pdist <scipy.spatial.distance.pdist>` as the `metric` argument.
Consequently, the distance data is stores as a distance matrix for all 
input locations passed to :class:`Variogram <skgstat.Variogram>` on 
instantiation. To be more precise, only the upper triangle is stored 
in an :class:`array <numpy.ndarray>` with the distance values sorted 
row-wise. Consider this very straightforward set of locations:

.. ipython:: python
    :okwarning:

    locations = [[0,0], [0,1], [1,1], [1,0]]
    V = Variogram(locations, [0, 1, 2, 1], normalize=False)

    V.distance

    # turn into a 2D matrix again
    from scipy.spatial.distance import squareform

    print(squareform(V.distance))


Binning
-------

As already mentioned, in real world observation data, there will hardly
be two observation location pairs at *exactly* the same distance. 
Thus, we need to group information about point pairs at *similar* distance
together, to learn how similar their observed values are. 
With a :class:`Variogram <skgstat.Variogram>`, we will basically try
to find and describe some systematic statistical behavior from these 
similarities. The process of grouping distance data together is 
called binning.

``scikit-gstat`` has two different methods for binning distance data. 
They can be set using the :func:`bin_func <skgstat.Variogram.bin_func>`
attribute. You have to pass the name of the method. 
This has to be one of ``['even', 'uniform]`` to use one of the predefined 
binning functions. Both methods will use two parameters to calculate the
bins from the distance matrix: ``n``, the amount of bins, 
and ``maxlag``, the maximum distance lag to be considered. You can choose
both parameters during Variogram instantiation as 
:func:`n_lags <skgstat.Variogram.n_lags>` and 
:func:`maxlag <skgstat.Variogram.maxlag>`. The ``'even'`` method will 
then form ``n`` bins from ``0`` to ``maxlag`` of same width. 
The ``'uniform'`` method will form ``n`` bins from ``0`` to ``maxlag`` 
with the same value count in each bin.
The following example should illustrate this:

.. ipython:: python
    :okwarning:

    from skgstat.binning import even_width_lags, uniform_count_lags
    from scipy.spatial.distance import pdist

    loc = np.random.normal(50, 10, size=(30, 2))
    distances = pdist(loc)


Now, look at the different bin edges for the calculated dummy 
distance matrix:

.. ipython:: python
    :okwarning: 

    even_width_lags(distances, 10, 250)
    uniform_count_lags(distances, 10, 250)



Observation differences
-----------------------

By the term *observation differences*, the distance between the 
observed values are meant. As already layed out, the main idea of 
a variogram is to systematially relate similarity of observations 
to their spatial proximity. The spatial part was covered in the 
sections above, finalized with the calculation of a suitable 
binning of all distances. We want to relate exactly these bins
to a measure of similarity of all observation point pairs that 
fall into this bin.

That's basically it. We need to do three more steps to come up 
with *one* value per bin, statistically describing the similarity
at that distance.

    1. Find all point pairs that fall into a bin
    2. Calculate the *distance* (difference) of the observed values
    3. Describe all differences by one number


Finding all pairs within a bin is straightforward. We already have 
the bin edges and all distances between all possible observation 
point combinations (stored in the distance matrix). Using the 
:func:`squareform <scipy.spatial.distance.squareform>` function 
of scipy, we *could* turn the distance matrix into a 2D version.
Then the row and column indices align with the values indices.
However, the :class:`Variogram Class <skgstat.Variogram>` implements 
a method for doing mapping a bit more efficiently.

.. note::

    As of this writing, the actual iterator that yields the group
    number for each point is written in a plain Python loop. 
    This is not very fast and in fact the main bottleneck of this class.
    I am evaluating numba, cython or a numpy based solution at the moment
    to gain better performance.

A :class:`array <numpy.ndarray>` of bin groups for each point pair that 
is indexed exactly like the :func:`distance <skgstat.Variogram.distance`
array can be obtained by :func:`lag_groups <skgstat.Variogram.lag_groups>`.

This will be illustrated by some sample data (you can find the CSV file 
in the github repository of SciKit-GStat).
You can easily read the data using pandas.

.. ipython:: python
    :okwarning:

    import pandas as pd 
    data = pd.read_csv('data/sample.csv')

    V = Variogram(list(zip(data.x, data.y)), data.z, 
        normalize=True, n_lags=25)
    
    @savefig variogram_sample_data.png width=8in
    V.plot()

Then, you can compare the first 10 point pairs from the distance matrix
to the first 10 elements returned by the 
:func:`lag_groups function <skgstat.Variogram.lag_groups>`.

.. ipython:: python
    :okwarning:

    # first 10 distances
    V.distance[:10]

    # first 10 groups
    V.lag_groups()[:10]

Now, we need the actual :func:`Variogram.bins <skgstat.Variogram.bins>` 
to verify the grouping.

.. ipython:: python 
    :okwarning:

    V.bins

The first and 9th element are grouped into group ``3``. Their values are
``20.8`` and ``18.8``. The grouping starts with ``0``, therefore the 
corresponding upper bound of the bin is at index ``3`` and the lower at 
``2``. The bin edges are therefore ``15.8 < x < 21.07``. 
Consequently, the binning and grouping worked fine.

If you want to access all value pairs at a given group, it would of 
course be possible to use the machanism above to find the correct points.
However, :class:`Variogram class <skgstat.Variogram>` offers an iterator 
that already does that for you: 
:func:`lag_classes <skgstat.Variogram.lag_classes>`. This iterator 
will yield all pair-wise observation value differences for the bin 
of the actual iteration. The first iteration (index = 0, if you wish) 
will yield all differences of group id ``0``. 

.. note::

    :func:`lag_classes <skgstat.Variogram.lag_classes>` will yield 
    the difference in value of observation point pairs, not the pairs 
    themselves.

.. ipython:: python

    for i, group in enumerate(V.lag_classes()):
        print('[Group %d]: %.2f' % (i, np.mean(group)))

The only thing that is missing for a variogram is that we will not 
use the arithmetic mean to describe the realtionship.

Experimental variograms
-----------------------

The last stage before a variogram function can be modeled is to define 
an empirical variogram, also known as *experimental variogram*, which
will be used to parameterize a variogram model.
However, the expermental variogram already contains a lot of information 
about spatial relationships in the data. Therefore, it's worth looking 
at more closely. Last but not least a poor expermental variogram will 
also affect the variogram model, which is ultimatively used to interpolate
the input data.

The previous sections summarized how distance is calculated and handeled 
by the :class:`Variogram class <skgstat.Variogram>`. 
The :func:`lag_groups function <skgstat.Variogram.lag_groups>` makes it 
possible to find corresponding observation value pairs for all distance 
lags. Finally the last step will be to use a more suitable estimator 
for the similarity of observation values at a specific lag. 
In geostatistics this estimator is called semi-variance and the
the most popular estimator is called *Matheron estimator*. 
In case the estimator used is not further specified, Matheron was used.
It is defined as 

.. math::
        \gamma (h) = \frac{1}{2N(h)} * \sum_{i=1}^{N(h)}(x)^2

with:

.. math::
    x = Z(x_i) - Z(x_{i+h})

where :math:`Z(x_i)` is the observation value at the i-th location 
:math:`x_i`. :math:`h` is the distance lag and :math:`N(h)` is the 
number of point pairs at that lag.

You will find more estimators in :mod:`skgstat.estimators`. 
There is the :func:`Cressie-Hawkins <skgstat.estimators.cressie>`, 
which is more robust to extreme values. Other so called robust 
estimators are :func:`Dowd <skgstat.estimators.dowd>` or 
:func:`Genton <skgstat.estimators.genton>`.
The remaining are experimental estimators and should only be used 
with caution. 

.. ipython:: python
    :okwarning:

    fig, _a = plt.subplots(2, 2, figsize=(8,8))
    axes = _a.flatten()

    V.plot(axes=axes[0], hist=False)
    V.estimator = 'cressie'
    V.plot(axes=axes[1], hist=False)
    V.estimator = 'dowd'
    V.plot(axes=axes[2], hist=False)
    V.estimator = 'genton'
    V.plot(axes=axes[3], hist=False)

    @savefig compare_estimators.png width=8in
    fig.show()

When direction matters
======================

What is 'direction'?
--------------------


Space-time variography
======================
