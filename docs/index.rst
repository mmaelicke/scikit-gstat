====
Home
====

Welcome to SciKit GStat's documentation!
========================================

`Download the docs as PDF <https://mmaelicke.github.io/scikit-gstat/SciKitGStat.pdf>`_

SciKit-Gstat is a scipy-styled analysis module for geostatistics. It includes
two base classes :class:`Variogram <skgstat.Variogram>` and
:class:`DirectionalVariogram <skgstat.DirectionalVariogram>`. Both have a
very similar interface and can compute experimental variograms and model
variograms. The module makes use of a rich selection of semi-variance
estimators and variogram model functions, while being extensible at the same
time.

With version 0.2.4, the class
:class:`SpaceTimeVariogram <skgstat.SpaceTimeVariogram>` has been added. It
computes space-time experimental variogram. However, space-time modeling is
not implemented yet.

With version 0.25, the class :class:`OrdinaryKriging <skgstat
.OridnaryKrigin>` has been added. It is working and can be used. However, it
is not documented, the arguments might still change, multiprocessing is not
implemented and the krige algorithm is not yet very efficient.

.. note::

    Scikit-gstat was rewritten in major parts. Most of the changes are internal,
    but the attributes and behaviour of the `Variogram` has also changed
    substantially.
    A detailed description of the new versions usage will follow. The last
    version of the old Variogram class, 0.1.8, is kept in the `version-0.1.8`
    branch on GitHub, but not developed any further. It is not compatible to
    the current version.

How to cite
===========

In case you use SciKit-GStat in other software or scientific publications,
please reference this module. It is published and has a DOI. It can be cited
as:

  Mirko Mälicke, & Helge David Schneider. (2019, November 7). Scikit-GStat 0.2.6: 
  A scipy flavoured geostatistical analysis toolbox written in Python. 
  (Version v0.2.6). Zenodo. http://doi.org/10.5281/zenodo.3531816

.. toctree::
    :maxdepth: 3
    :caption: Contents:

    install
    getting_started
    userguide/userguide
    tutorials/tutorials
    technical/technical
    reference/reference
    changelog
    PDF <https://mmaelicke.github.io/scikit-gstat/SciKitGStat.pdf>

