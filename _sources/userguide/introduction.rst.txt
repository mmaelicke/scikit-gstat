============
Introduction
============

General
=======

This user guide part of ``scikit-gstat``'s documentation is meant to be an
user guide to the functionality offered by the module along with a more
general introduction to geostatistical concepts. The main use case is to hand
this description to students learning geostatistics, whenever
``scikit-gstat`` is used.
But before introducing variograms, the more general question what
geostatistics actually are has to be answered.

.. note::

    This user guide is meant to be an **introduction** to geostatistics. In
    case you are already familiar with the topic, you can skip this section.

What is geostatistics?
======================

The basic idea of geostatistics is to describe and estimate spatial
correlations in a set of point data. While the main tool, the variogram, is
quite easy to implement and use, a lot of assumptions are underlying it.
The typical application is geostatistics is an interpolation. Therefore,
although using point data, a basic concept is to understand these point data
as a sample of a (spatially) continuous variable that can be described as a
random field :math:`rf`, or to be more precise, a Gaussian random field in many
cases. The most fundamental assumption in geostatistics is that any two values
:math:`x_i` and :math:`x_{i + h}` are more similar, the smaller :math:`h` is,
which is a separating distance on the random field. In other words: *close
observation points will show higher covariances than distant points*. In case
this most fundamental conceptual assumption does not hold for a specific
variable, geostatistics will not be the correct tool to analyse and
interpolate this variable.

One of the most easiest approaches to interpolate point data is to use IDW
(inverse distance weighting). This technique is implemented in almost any GIS
software. The fundamental conceptual model can be described as:

.. math::
    Z_u = \frac{\sum_{i}^{N} w_i * Z(i)}{N}

where :math:`Z_u` is the value of :math:`rf` at a non-observed location with
:math:`N` observations around it. These observations get weighted by the weight
:math:`w_i`, which can be calculated like:

.. math::
    w_i = \frac{1}{||\overrightarrow{ux_i}||}

where :math:`u` is the not observed point and :math:`x_i` is one of the
sample points. Thus, :math:`||\overrightarrow{ux_i}||` is the 2-norm of
the vector between the two points: the Euclidean distance in the coordinate
space (which by no means has to be limited to the :math:`\mathbb{R}^2` case).

This basically describes a concept, where a value of the random field is
estimated by a distance-weighted mean of the surrounding points. As close
points shall have a higher impact, the inverse distance is used and thus the
name of **inverse distance weighting**.

In the case of geostatistics this basic model still holds, but is extended.
Instead of depending the weights exclusively on the separating distance, a
weight will be derived from a variance over all values that are separated by
a similar distance. This has the main advantage of incorporating the actual
(co)variance found in the observations and basing the interpolation on this
(co)variance, but comes at the cost of some strict assumptions about the
statistical properties of the sample. Elaborating and assessing these
assumptions is one of the main challenges of geostatistics.

Geostatistical Tools
====================

Geostatistics is a wide field spanning a wide variety of disciplines, like
geology, biology, hydrology or geomorphology. Each discipline defines their
own set of tools, and apparently definitions, and progress is made until
today. It is not the objective of ``scikit-gstat`` to be a comprehensive
collection of all available tools. That would only be possible if
professionals from each discipline contribute to the project. The objective
is more to offer some common tools and simplify the process of geostatistical
analysis and tool development thereby.
However, you split geostatistics into three main fields, each of it with its
own tools:

* **variography:** with the variogram being the main tool, the variography
  focuses on describing, visualizing and modelling covariance structures in
  space and time.
* **kriging:** is an interpolation method, that utilizes a variogram to find
  the estimate for weights as shown in the section above.
* **geostatistical simulation:** is aiming on generate random fields that fit
  a given set of observations or a pre-defined variogram.

.. note::

    I am planning to implement common tools from all three fields. However,
    up to now, I am only focusing on variograms and no field generators or
    kriging procedures are available.


How to use this Guide
=====================

*Write something about code examples and stuff*