====
Home
====
.. SciKit GStat documentation master file, created by
   sphinx-quickstart on Mon Apr 30 14:57:02 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SciKit GStat's documentation!
========================================


At current state, this module offers a scipy-styled `Variogram` class for
performing geostatistical analysis. This class can be used to derive
variograms. Key benefits are a number of semivariance estimators and theoretical
variogram functions. The module is planned to be hold in the manner of scikit
 modules and be based upon `numpy` and `scipy` whenever possible. There is
 also a distance matrix extension available, with a function for calculating
n-dimensional distance matrices for the variogram.

.. note::

    Scikit-gstat was rewritten in major parts. Most of the changes are internal,
    but the attributes and behaviour of the `Variogram` has also changed
    substantially.
    A detailed description of of the new versions usage will follow. The last
    version of the old Variogram class, 0.1.8, is kept in the `version-0.1.8`
    branch on GitHub, but not developed any further. Those two versions are not
    compatible.


.. toctree::
    :maxdepth: 3
    :caption: Contents:

    install
    getting_started
    reference/reference

