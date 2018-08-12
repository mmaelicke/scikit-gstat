Scikit-Gstat
============

Info: scikit-gstat needs Python >= 3.4!

.. image:: https://travis-ci.org/mmaelicke/scikit-gstat.svg?branch=master
    :target: https://travis-ci.org/mmaelicke/scikit-gstat
    :alt: Build Status

.. image:: https://readthedocs.org/projects/scikit-gstat/badge/?version=latest
    :target: http://scikit-gstat.readthedocs.io/en/latest?badge=latest
    :alt: Documentation Status

.. image:: https://codecov.io/gh/mmaelicke/scikit-gstat/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/mmaelicke/scikit-gstat
    :alt: Codecov


New Version 0.2
~~~~~~~~~~~~~~~

Scikit-gstat was rewritten in major parts. Most of the changes are internal,
but the attributes and behaviour of the `Variogram` has also changed
substantially.
A detailed description of of the new versions usage will follow. The last
version of the old Variogram class, 0.1.8, is kept in the `version-0.1.8`
branch on GitHub, but not developed any further. Those two versions are not
compatible.

Description
~~~~~~~~~~~
At current state, this module offers a scipy-styled `Variogram` class for performing geostatistical analysis.
This class can be used to derive variograms. Key benefits are a number of semivariance estimators and theoretical
variogram functions. The module is planned to be hold in the manner of scikit modules and be based upon `numpy` and
`scipy` whenever possible. There is also a distance matrix extension available, with a function for calculating
n-dimensional distance matrices for the variogram.
The estimators include:

- matheron
- cressie
- dowd
- genton
- entropy
- two experimental ones: quantiles, minmax

The models include:

- sperical
- exponential
- gaussian
- cubic
- stable
- matérn

with all of them in a nugget and no-nugget variation. All the estimator are
implemented using numba's jit decorator. The usage of numba might be subject
to change in future versions.
At the current stage, the package does not include any kriging. This is planned for a future release.


Installation
~~~~~~~~~~~~

PyPI:

.. code-block:: bash

  pip install scikit-gstat

GIT:

.. code-block:: bash

  git clone https://github.com/mmaelicke/scikit-gstat.git
  cd scikit-gstat
  pip install -r requirements.txt
  pip install -e .

Usage
~~~~~

The `Variogram` class needs at least a list of coordiantes and values. All other attributes are set by default.
You can easily set up an example by generating some random data:

.. code-block:: python

  import numpy as np
  import skgstat as skg

  coordinates = np.random.gamma(0.7, 2, (30,2))
  values = np.random.gamma(2, 2, 30)

  V = skg.Variogram(coordinates=coordinates, values=values)
  print(V)

.. code-block:: bash

  spherical Variogram
  -------------------
  Estimator:    matheron
  Range:        1.64
  Sill:         5.35
  Nugget:       0.00