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
    from pprint import pprint

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
    data = pd.read_csv('data/sample_sr.csv')

    V = Variogram(list(zip(data.x, data.y)), data.z, 
        normalize=False, n_lags=25, maxlag=60)
    
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

Variogram models
----------------

The last step to describe the spatial pattern in a data set 
using variograms is to model the empirically observed and calculated
experimental variogram with a proper mathematical function. 
Technically, this setp is straightforward. We need to define a 
function that takes a distance value (not a lag) and returns 
a semi-variance value. One big advantage of these models is, that we 
can assure different things, like positive definitenes. Most models
are also monotonically increasing and approach an upper bound.
Usually these models need three parameters to fit to the experimental
variogram. All three parameters have a meaning and are usefull
to learn something about the data. This upper bound a model approaches
is called *sill*. The distance at which 95% of the sill are approached 
is called the *range*. That means, the range is the distance at which 
observation values do **not** become more dissimilar with increasing 
distance. They are statistically independent. That also means, it doesn't 
make any sense to further describe spatial relationships of observations 
further apart with means of geostatistics. The last parameter is the *nugget*.
It is used to add semi-variance to all values. Graphically that means to
*move the variogram up on the y-axis*. The nugget is the semi-variance modeled
on the 0-distance lag. Compared to the sill it is the share of variance that
can not be described spatially.

The spherical model
~~~~~~~~~~~~~~~~~~~

The sperical model is the most commonly used variogram model. 
It is characterized by a very steep, exponential increase in semi-variance.
That means it approaches the sill quite quickly. It can be used when 
observations show strong dependency on short distances.
It is defined like:

.. math::
    \gamma = b + C_0 * \left({1.5*\frac{h}{r} - 0.5*\frac{h}{r}^3}\right)

if h < r, and

.. math::
    \gamma = b + C_0
    
else. ``b`` is the nugget, :math:``C_0`` is the sill, ``h`` is the input
distance lag and ``r`` is the effective range. That is the range parameter 
described above, that describes the correlation length. 
Many other variogram model implementations might define the range parameter, 
which is a variogram parameter. This is a bit confusing, as the range parameter 
is specific to the used model. Therefore I decided to directly use the 
*effective range* as a parameter, as that makes more sense in my opinion. 
 
As we already calculated an experimental variogram and find the spherical 
model in the :py:mod:`skgstat.models` sub-module, we can utilize e.g. 
:func:`curve_fit <scipy.optimize.curve_fit>` from scipy to fit the model 
using a least squares approach.
 
.. ipython:: python
    :okwarning:
 
    from skgstat import models

    # set estimator back
    V.estimator = 'matheron'
    V.model = 'spherical'

    xdata = V.bins
    ydata = V.experimental
   
    from scipy.optimize import curve_fit
    
    cof, cov =curve_fit(models.spherical, xdata, ydata)
    
Here, *cof* are now the coefficients found to fit the model to the data.

.. ipython:: python
    :okwarning:

    pprint("range: %.2f\nsill: %.f\nnugget: %.2f" % (cof[0], cof[1], cof[2]))
  
.. ipython:: python
    :okwarning:
    
    xi =np.linspace(xdata[0], xdata[-1], 100)
    yi = [models.spherical(h, *cof) for h in xi]
    
    plt.plot(xdata, ydata, 'og')
    @savefig manual_fitted_variogram.png width=8in
    plt.plot(xi, yi, '-b');

The :class:`Variogram Class <skgstat.Variogram>` does in principle the 
same thing. The only difference is that it tries to find a good 
initial guess for the parameters and limits the search space for 
parameters. That should make the fitting more robust. 
Technically, we used the Levenberg-Marquardt algorithm above. 
:class:`Variogram <skgstat.Variogram>` can be forced to use the same 
by setting the :class:`Variogram.fit_method <skgstat.Variogram.fit_method>` 
to 'lm'. The default, however, is 'trf', which is the *Trust Region Reflective* 
algorithm, the bounded fit with initial guesses described above.
You can use it like:

.. ipython:: python
    :okwarning:

    V.fit_method ='trf'
    @savefig trf_automatic_fit.png width=8in
    V.plot();
    pprint(V.describe())
    
    V.fit_method ='lm'
    @savefig lm_automatic_fit.png width=8in
    V.plot();
    pprint(V.describe())

.. note::

    In this example, the fitting method does not make a difference 
    at all. Generally, you can say that Levenberg-Marquardt is faster
    and TRF is more robust.

Exponential model
~~~~~~~~~~~~~~~~~

The exponential model is quite similar to the spherical one. 
It models semi-variance values to increase exponentially with 
distance, like the spherical. The main difference is that this 
increase is not as steep as for the spherical. That means, the 
effective range is larger for an exponential model, that was 
parameterized with the same range parameter.

.. note::

    Remember that SciKit-GStat uses the *effective range* 
    to overcome this confusing behaviour.

Consequently, the exponential can be used for data that shows a way
too large spatial correlation extent for a spherical model to 
capture. 

Applied to the data used so far, you can see the similarity between 
the two models:

.. ipython:: python
    :okwarning:

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    V.fit_method = 'trf'
    V.plot(axes=axes[0], hist=False)

    V.model = 'exponential'
    @savefig compare_spherical_exponential.png width=8in
    V.plot(axes=axes[1], hist=False);

Gaussian model
~~~~~~~~~~~~~~

The last fundamental variogram model is the Gaussian. 
Unlike the spherical and exponential models a very different 
spatial relationship between semi-variance and distance.
Following the Gaussian model, observations are assumed to 
be similar up to intermediate distances, showing just a 
gentle increase in semi-variance. Then, the semi-variance 
increases dramatically wihtin just a few distance units up 
to the sill, which is again approached asymtotically.
The model can be used to simulate very sudden and sharp 
changes in the variable at a specific distance, 
while being very similar at smaller distances.

To show a typical Gaussian model, we will load another 
sample dataset.

.. ipython:: python
    :okwarning:

    data = pd.read_csv('data/sample_lr.csv')

    Vg = Variogram(list(zip(data.x, data.y)), data.z.values,
        normalize=False, n_lags=25, maxlag=90, model='gaussian')

    @savefig sample_data_gaussian_model.png width=8in
    Vg.plot();

Matérn model
~~~~~~~~~~~~

One of the not so commonly used models is the Matérn model. 
It is nevertheless implemented into scikit-gstat as it is one 
of the most powerful models. Especially in cases where you cannot 
chose the appropiate model a priori so easily.
The Matérn model takes an additional smoothness paramter, that can 
change the shape of the function in between an exponential 
model shape and a Gaussian one. 

.. ipython:: python
    :okwarning:
   
    xi = np.linspace(0, 100, 100)

    # plot a exponential and a gaussian
    y_exp = [models.exponential(h, 40, 10, 3) for h in xi]
    y_gau = [models.gaussian(h, 40, 10, 3) for h in xi]

    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    ax.plot(xi, y_exp, '-b', label='exponential')
    ax.plot(xi, y_gau, '-g', label='gaussian')

    for s in (0.1, 2., 10.):
        y = [models.matern(h, 40, 10, 3, s) for h in xi]
        ax.plot(xi, y, '--k', label='matern s=%.1f' % s)
    @savefig compare_smoothness_parameter_matern.png width=8in
    plt.legend(loc='lower right')

When direction matters
======================

What is 'direction'?
--------------------

The classic approach to calculate a variogram is based on the 
assumption that covariance between observations can be related to 
their separating distance. For this, point pairs of all observation 
points are formed and it is assumed that they can be formed without any restriction.
The only paramter to be influenced is a limiting distance, beyond which 
a point pair does not make sense anymore. 

This assumption might not always hold. Especially in landscapes, processes do 
not occur randomly, but in an organized manner. This organization is often 
directed, which can lead to stronger covariance in one direction than another.
Therefore, another step has to be introduced before lag classes are formed.

The *direction* of a variogram is then a orientation, which two points need. 
If they are not oriented in the specified way, they will be ignored while calculating 
a semi-variance value for a given lag class. Usually, you will specify a 
orientation, which is called :func:`azimuth <skgstat.DirectionalVariogram.azimuth>`, 
and a :func:`tolerance <skgstat.DirectionalVariogram.tolerance>`, which is an 
offset from the given azimuth, at which a point pair will still be accepted.

Defining orientiation
---------------------

One has to decide how orientation of two points is determined. In scikit-gstat,
orientation between two observation points is only defined in :math:`\mathbb{R}^2`.
We define the orientation as the **angle between the vector connecting two observation points 
with the x-axis**.

Thus, also the :func:`azimuth <skgstat.DirectionalVariogram.azimuth>` is defined as an 
angle of the azimutal vector to the x-axis, with an 
:func:`tolerance <skgstat.DirectionalVariogram.tolerance>` in degrees added to the 
exact azimutal orientation clockwise and counter clockwise.

The angle :math:`\Phi` between two vetors ``u,v`` is given like:

.. math::

    \Phi = cos^{-1}\left(\frac{u \circ v}{||u|| \cdot ||v||}\right)

.. ipython:: python
    :okwarning:

    from matplotlib.patches import FancyArrowPatch as farrow
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    ax.arrow(0,0,2,1,color='k')
    ax.arrow(-.1,0,3.1,0,color='k')
    ax.set_xlim(-.1, 3)
    ax.set_ylim(-.1,2.)
    ax.scatter([0,2], [0,1], 50, c='r')
    ax.annotate('A (0, 0)', (.0, .26), fontsize=14)
    ax.annotate('B (2, 1)', (2.05,1.05), fontsize=14)
    arrowstyle="Simple,head_width=6,head_length=12,tail_width=1"
    ar = farrow([1.5,0], [1.25, 0.625],  color='r', connectionstyle="arc3, rad=.2", arrowstyle=arrowstyle)
    ax.add_patch(ar)
    @savefig sample_orientation_of_2_1.png width=6in
    ax.annotate('26.5°', (1.5, 0.25), fontsize=14, color='r')

The described definition of orientation is illustrated in the figure above. 
There are two observation points, :math:`A (0,0)` and :math:`B (2, 1)`. To decide
wether to account for them when calculating the semi-variance at their separating 
distance lag, their orientation is used. Only if the direction of the varigram includes
this orientation, the points are used. Imagine the azimuth and tolerance would be 
``45°``, then anything between ``0°`` (East) and ``90°`` orientation would be included.
The given example shows the orientation angle :math:`\Phi = 26.5°`, which means the 
vector :math:`\overrightarrow{AB}` is included.

Calculating orientations
------------------------

SciKit-GStat implements a slightly adaped version of the formula given in the 
last section. It makes use of symmetric search areas (tolerance is applied clockwise 
and counter clockwise) und therefore any calculated angle might be the result 
of calculating the orientation of :math:`\overrightarrow{AB}` or 
:math:`\overrightarrow{BA}`. Mathematically, these two vectors have two different 
angles, but they are always both taken into account or omitted for a variagram 
at the same time. Thus, it does not make a difference for variography. 
However, it does make a difference when you try to use the orientation angles 
directly as the containing matrix can contain the inverse angles.

This can be demonstrated by an easy example. Let ``c`` be a set of points mirrored 
along the x-axis.

.. ipython:: python
    :okwarning:

    c = np.array([[0,0], [2,1], [1,2], [2, -1], [1, -2]])
    east = np.array([1,0])

We can plug these two arrays into the the formula above:

.. ipython:: python
    :okwarning:

    u = c[1:]   # omit the first one
    angles = np.degrees(np.arccos(u.dot(east) / np.sqrt(np.sum(u**2, axis=1))))
    angles.round(1)

You can see, that the both points and their mirrored counterpart have the same 
angle to the x-axis, just like expected. This can be visualized by the plot below:

.. ipython:: python
    :okwarning:

    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    ax.set_xlim(-.1, 2.25)
    ax.set_ylim(-2.1,2.1)
    ax.arrow(-.1,0,3.1,0,color='k')
    for i,p in enumerate(u):
        ax.arrow(0,0,p[0],p[1],color='r')
        ax.annotate('%.1f°' % angles[i], (p[0] / 2, p[1] / 2 ), fontsize=14, color='r')
    @savefig sample_orientation_of_multiple_points.png width=6in
    ax.scatter(c[:,0], c[:,1], 50, c='r')

The main difference to the internal structure storing the orientation angles for a 
:class:`DirectionalVariogram <skgstat.DirectionalVariogram>` instance will store different
angles.
To use the class on only five points, we need to prevent the class from fitting, as 
fitting on only 5 points will not work. But this does not affect the orientation calculations.
Therefore, the :func:`fit <skgstat.DirectionalVariogram.fit>` mehtod is overwritten.

.. ipython:: python
    :okwarning:

    class TestCls(DirectionalVariogram):
        def fit(*args, **kwargs):
            pass

    DV = TestCls(c, np.random.normal(0,1,len(c)))
    DV._calc_direction_mask_data()
    np.degrees(DV._angles + np.pi)[:len(c) - 1]

The first two points (with positive y-coordinate) show the same result. The other two, 
with negative y-coordinates, are also calculated counter clockwise:

.. ipython:: python
    :okwarning:

    360 - np.degrees(DV._angles + np.pi)[[2,3]]

The :class:`DirectionalVariogram <skgstat.DirectionalVariogram>` class has a plotting 
function to show a network graph of all point pairs that are oriented in the 
variogram direction. But first we need to increase the tolerance as half tolerance 
(``45° / 2 = 22.5°`` clockwise and counter clockwise) is smaller than both orientations.

.. ipython:: python
    :okwarning:

    DV.tolerance = 90 
    @savefig sample_pair_field_plot.png width=8in
    DV.pair_field()

Directional variogram
---------------------


.. ipython:: python
    :okwarning:

    field = np.loadtxt('data/aniso_x2.txt')
    np.random.seed(1312)
    coords = np.random.randint(100, size=(300,2))
    vals = [field[_[0], _[1]] for _ in coords]

The next step is to create two different variogram instances, which share the same 
parameters, but use a different azimuth angle. One oriented to North and the 
second one oriented to East.

.. ipython:: python
    :okwarning:

    Vnorth = DirectionalVariogram(coords, vals, azimuth=90, tolerance=90, maxlag=80, n_lags=20)
    Veast = DirectionalVariogram(coords, vals, azimuth=0, tolerance=90, maxlag=80, n_lags=20)
    pd.DataFrame({'north':Vnorth.describe(), 'east': Veast.describe()})

You can see, how the two are differing in effective range and also sill, only 
caused by the orientation. Let's look at the experimental variogram:

.. ipython:: python

    fix, ax = plt.subplots(1,1,figsize=(8,6))
    ax.plot(Vnorth.bins, Vnorth.experimental, '.--r', label='North-South')
    ax.plot(Veast.bins, Veast.experimental, '.--b', label='East-West')
    ax.set_xlabel('lag [m]')
    ax.set_ylabel('semi-variance (matheron)')
    @savefig expermiental_direcional_varigram_comparison.png width=8in
    plt.legend(loc='upper left')

The shape of both experimental variograms is very similar on the first 40 meters 
of distance. Within this range, the apparent anisotropy is not pronounced. 
The East-West oriented variograms also have an effective range of only about 40 meters,
which means that in this direction the observations become statistically independent 
at larger distances.
For the North-South variogram the effective range is way bigger and the variogram 
plot reveals much larger correlation lengths in that direction. The spatial 
dependency is thus directed in North-South direction.


To perform Kriging, you would now transform the data, especially in North-West 
direction, unitl both variograms look the same within the effective range. 
Finally, the Kriging result is back-transformed into the original coordinate system.
