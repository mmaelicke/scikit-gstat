====
Home
====

Welcome to SciKit GStat
=======================

`Download the docs as PDF <https://mmaelicke.github.io/scikit-gstat/SciKitGStat.pdf>`_

.. image:: https://img.shields.io/pypi/v/scikit-gstat?color=green&logo=pypi&logoColor=yellow&style=flat-square   :alt: PyPI
    :target: https://pypi.org/project/scikit-gstat

.. image:: https://img.shields.io/github/v/release/mmaelicke/scikit-gstat?color=green&logo=github&style=flat-square   :alt: GitHub release (latest by date)
    :target: https://github.com/mmaelicke/scikit-gstat

.. image:: https://github.com/mmaelicke/scikit-gstat/workflows/Test%20and%20build%20docs/badge.svg
    :target: https://github.com/mmaelicke/scikit-gstat/actions

.. image:: https://api.codacy.com/project/badge/Grade/34022fb8b795435b8eeb5431159fa7c6
   :alt: Codacy Badge
   :target: https://app.codacy.com/app/mmaelicke/scikit-gstat?utm_source=github.com&utm_medium=referral&utm_content=mmaelicke/scikit-gstat&utm_campaign=Badge_Grade_Dashboard

.. image:: https://codecov.io/gh/mmaelicke/scikit-gstat/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/mmaelicke/scikit-gstat
    :alt: Codecov

.. image:: https://zenodo.org/badge/98853365.svg
   :target: https://zenodo.org/badge/latestdoi/98853365

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

  Mirko Mälicke, Helge David Schneider, Sebastian Müller, & Egil Möller. (2021, April 20). 
    mmaelicke/scikit-gstat: A scipy flavoured geostatistical variogram analysis toolbox 
    (Version v0.5.0). Zenodo. http://doi.org/10.5281/zenodo.4704356

.. note::

    Scikit-gstat was rewritten in major parts. Most of the changes are internal,
    but the attributes and behaviour of the `Variogram` has also changed
    substantially.
    A detailed description of the new versions usage will follow. The last
    version of the old Variogram class, 0.1.8, is kept in the `version-0.1.8`
    branch on GitHub, but not developed any further. It is not compatible to
    the current version.

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

