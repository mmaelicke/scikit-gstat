====
Home
====

Welcome to SciKit GStat
=======================

`Download the docs as PDF <https://mmaelicke.github.io/scikit-gstat/SciKitGStat.pdf>`_


SciKit-Gstat is a scipy-styled analysis module for variogram analysis. 
The base class is called :class:`Variogram <skgstat.Variogram>`, which is probably the
only import needed. However, several other classes exist:

* :class:`DirectionalVariogram <skgstat.DirectionalVariogram>` for directional variograms
* :class:`SpaceTimeVariogram <skgstat.SpaceTimeVariogram>` for spatio-temporal variograms
* :class:`OrdinaryKriging <skgstat.OrdinaryKriging>` for interpolation
* :class:`MetricSpace <skgstat.MetricSpace>` for pre-computed spatial samples

The variogram classes have a similar interface and can compute experimental variograms 
and fit theoretical variogram model functions. 
The module makes use of a rich selection of semi-variance estimators, variogram model functions
and sptial binning functions, while being extensible at the same time.

How to cite
===========

In case you use SciKit-GStat in other software or scientific publications,
please reference this module. It is published and has a DOI. It can be cited
as:

  Mirko Mälicke, Egil Möller, Helge David Schneider, & Sebastian Müller. (2021, May 28). 
    mmaelicke/scikit-gstat: A scipy flavoured geostatistical variogram analysis toolbox 
    (Version v0.6.0). Zenodo. http://doi.org/10.5281/zenodo.4835779


.. toctree::
    :maxdepth: 3
    :caption: Contents:

    install
    getting_started
    userguide/userguide
    auto_examples/index
    technical/technical
    reference/reference
    changelog
    PDF <https://mmaelicke.github.io/scikit-gstat/SciKitGStat.pdf>

