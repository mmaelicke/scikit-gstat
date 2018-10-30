======================
Directional Variograms
======================

General
=======

With version 0.2.2, directional variograms have been introduced. A
directional variogram is a variogram where point pairs are only included into
the semivariance calculation if they fulfill a specified spatial relation.
This relation is expressed as a *search area* that identifies all
*directional* points for a given specific point. SciKit-GStat refers to this
point as *poi* (point of interest). The implementation is done by the
:class:`DirectionalVariogram <skgstat.DirectionalVariogram>` class.

Understanding Search Area
=========================

.. note::

    The :class:`DirectionalVariogram <skgstat.DirectionalVariogram>` is
    in general capable of handling n-dimensional coordinates. The application
    of directional dependency is, however, only applied to the first two
    dimensions.

Understanding the search area of a directional is vital for using the
:class:`DirectionalVariogram <skgstat.DirectionalVariogram>` class. The
search area is controlled by the
:func:`directional_model <skgstat.DirectionalVariogram.directional_model>`
property which determines the shape of the search area. The extend and
orientation of this area is controlled by the parameters:

- :func:`azimuth <skgstat.DirectionalVariogram.azimuth>`
- :func:`tolerance <skgstat.DirectionalVariogram.tolerance>`
- :func:`bandwidth <skgstat.DirectionalVariogram.bandwidth>`

As of this writing, SciKit-GStat supports three different search area shapes:

- :func:`triangle <skgstat.DirectionalVariogram._triangle>` (*default*)
- :func:`circle <skgstat.DirectionalVariogram._circle>`
- :func:`compass <skgstat.DirectionalVariogram._compass>`

Additionally, the shape generation is controlled by the
:func:`tolerance <skgstat.DirectionalVariogram.tolerance>` parameter
(:func:`triangle <skgstat.DirectionalVariogram._triangle>`,
:func:`compass <skgstat.DirectionalVariogram._compass>`) and
:func:`bandwidth <skgstat.DirectionalVariogram.bandwidth>` parameter
(:func:`triangle <skgstat.DirectionalVariogram._triangle>`,
:func:`circle <skgstat.DirectionalVariogram._circle>`). The
:func:`azimuth <skgstat.DirectionalVariogram.azimuth>` is used to rotate the
search area into a desired direction. An azimuth of 0° is heading East of the
coordinate plane. Positive values for azimuth rotate the search area
clockwise, negative values counter-clockwise.
The :func:`tolerance <skgstat.DirectionalVariogram.tolerance>` specifies how
far the angle (against 'x-axis') between two points can be off the azimuth to
be still considered as a directional point pair. Based on this definition,
two points at a larger distance would generally be allowed to differ more
from azimuth in terms of coordinate distance. Therefore the
:func:`bandwidth <skgstat.DirectionalVariogram.bandwidth>` defines a maximum
coordinate distance, a point can have from the azimuth line.
The difference between the
:func:`triangle <skgstat.DirectionalVariogram._triangle>` and the
:func:`compass <skgstat.DirectionalVariogram._compass>` search area is that
the triangle uses the bandwidth and the compass does not.

The :class:`DirectionalVariogram <skgstat.DirectionalVariogram>` has a
function to plot the current search area. As the search area is specific to
the current poi, it has to be defined as the index of the coordinate to be used.
The method is called
:func:`search_area <skgstat.DirectionalVariogram.search_area>`.
Using random coordinates, the search area shapes are presented below.

.. ipython:: python
    :okwarning:

    from skgstat import DirectionalVariogram
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    np.random.seed(42)
    coords = np.random.gamma(15, 6, (40, 2))
    np.random.seed(42)
    vals = np.random.normal(5,1, 40)

    DV = DirectionalVariogram(coords, vals,
        azimuth=0,
        tolerance=45,
        directional_model='triangle')

    @savefig dv1.png width=6in
    DV.search_area(poi=3)

The model can easily be changed, using the
:func:`set_directional_model <skgstat.DirectionalVariogram.set_directional_model>`
function:

.. ipython:: python
    :okwarning:

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    DV.set_directional_model('circle')
    DV.search_area(poi=3, ax=axes[0])

    DV.set_directional_model('compass')
    DV.search_area(poi=3, ax=axes[1])

    @savefig dv2.png width=8in
    fig.show()


Local Reference System
======================

In order to apply different search area shapes and rotate them considering
the given azimuth, a few preprocessing steps are necessary. This can lead to
some calculation overhead, as in the case of a
:func:`compass <skgstat.DirectionalVariogram._compass>` model.
The selection of point pairs being directional is implemented by transforming
the coordinates into a local reference system iteratively for each coordinate.
For multidimensional coordinates, only the first two dimensions are used.
They are shifted to make the current point of interest the origin of the
local reference system. Then all other points are rotated until the azimuth
overlays the local x-axis. This makes the definition of different shapes way
easier.
In this local system, the bandwidth can easily be applied to the transformed
y-axis. The
:func:`set_directional_model <skgstat.DirectionalVariogram.set_directional_model>`
can also set a custom function as search area shape, that accepts the current
local reference system and returns the search area for the given poi.
The search area has to be returned as a shapely Polygon. Unfortunately, the
:func:`tolerance <skgstat.DirectionalVariogram.tolerance>` and
:func:`bandwidth <skgstat.DirectionalVariogram.bandwidth>` parameter are not
passed yet.

.. note::

    For the next release, it is planned to pass all necessary parameters to
    the directional model function. This should greatly improve the
    definition of custom shapes. Until the implementation, the parameters
    have to be injected directly.

The following example will illustrate the rotation of the local reference
system.

.. ipython:: python
    :okwarning:

    from matplotlib.patches import FancyArrowPatch as arrow
    np.random.seed(42)
    c = np.random.gamma(10, 6, (9, 2))
    mid = np.array([[np.mean(c[:,0]), np.mean(c[:,1])]])
    coords = np.append(mid, c, axis=0) - mid

    DV = DirectionalVariogram(coords, vals[:10],
        azimuth=45, tolerance=45)

    loc = DV.local_reference_system(poi=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(coords[:,0], coords[:,1], 15, c='b')
    ax.scatter(loc[:,0], loc[:,1], 20, c='r')
    ax.scatter([0], [0], 40, c='y')

    for u,v in zip(coords[1:], loc[1:]):
        arrowstyle="Simple,head_width=6,head_length=12,tail_width=1"
        a = arrow(u, v,  color='grey', linestyle='--',
            connectionstyle="arc3, rad=.3", arrowstyle=arrowstyle)
        ax.add_patch(a)

    @savefig transform.png width=6in
    fig.show()

After moving and shifting, any type of Geometry could be generated and passed
as the search area.

.. ipython:: python
    :okwarning:

    from shapely.geometry import Polygon

    def M(loc):
        xmax = np.max(loc[:,0])
        ymax = np.max(loc[:,1])
        return Polygon([
            (0, 0),
            (0, ymax * 0.6),
            (0.05*xmax, ymax * 0.6),
            (xmax * 0.3, ymax * 0.3),
            (0.55 * xmax, 0.6 * ymax),
            (0.6 * xmax, 0.6 * ymax),
            (0.6 * xmax, 0),
            (0.55 * xmax, 0),
            (0.55 * xmax, 0.55 * ymax),
            (xmax * 0.325, ymax * 0.275),
            (xmax * 0.275, ymax * 0.275),
            (0.05 * xmax, 0.55 * ymax),
            (0.05 * xmax, 0),
            (0, 0)
        ])

    DV.set_directional_model(M)

    @savefig custom.png width=6in
    DV.search_area(poi=0)


Directional variograms
======================

In principle, the :class:`DirectionalVariogram <skgstat.DirectionalVariogram>`
can be used just like the :class:`Variogram <skgstat.Variogram>` base class.
In fact :class:`DirectionalVariogram <skgstat.DirectionalVariogram>` inherits
most of the behaviour. All the functionality described in the previous
sections is added to the basic :class:`Variogram <skgstat.Variogram>`.
All other methods and attributes can be used in the same way.

.. warning::

    In order to implement the directional dependency, some methods have been
    rewritten in :class:`DirectionalVariogram <skgstat.DirectionalVariogram>`.
    Thus the following methods do **not** show the same behaviour:

    - :func:`DirectionalVariogram.bins <skgstat.DirectionalVariogram.bins>`
    - :func:`DirectionalVariogram._calc_groups <skgstat.DirectionalVariogram._calc_groups>`

