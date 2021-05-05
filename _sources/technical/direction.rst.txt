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
function to plot the effect of the search area. The method is called
:func:`pair_field <skgstat.DirectionalVariogram.pair_field>`. Using
random coordinates, the visualization is shown below.

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
    DV.pair_field(plt.gca())
    
The model can easily be changed, using the
:func:`set_directional_model <skgstat.DirectionalVariogram.set_directional_model>`
function:

.. ipython:: python
    :okwarning:

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    DV.set_directional_model('triangle')
    DV.pair_field(plt.gca())

    @savefig dv2.png width=8in
    DV.pair_field(plt.gca())
    fig.show()

    DV.set_directional_model('compass')

    @savefig dv3.png width=8in
    DV.pair_field(plt.gca())
    fig.show()

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

