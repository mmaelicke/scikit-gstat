Scikit-Gstat
============

Info: scikit-gstat needs Python >= 3.5!

.. image:: https://travis-ci.org/mmaelicke/scikit-gstat.svg?branch=dev
    :target: https://travis-ci.org/mmaelicke/scikit-gstat
    :alt: Build Status

.. image:: https://readthedocs.org/projects/scikit-gstat/badge/?version=latest
    :target: http://scikit-gstat.readthedocs.io/en/latest?badge=latest
    :alt: Documentation Status

.. image:: https://codecov.io/gh/mmaelicke/scikit-gstat/branch/dev/graph/badge
.svg
    :target: https://codecov.io/gh/mmaelicke/scikit-gstat
    :alt: Codecov


New Version 0.2
~~~~~~~~~~~~~~~

Scikit-gstat is rewritten in major part at the moment. Most of the changes
are internal, but the usage of the `Variogram` class will also change. Once
merged into the master branch, a description of changes will follow. The last
Version of the current master branch, 0.1.7, will be kept in a new branch,
bu not any further developed in the future.

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
- dowd (not re-implemented)
- genton  (not re-implemented)
- entropy  (not re-implemented)
- bin quantiles  (not re-implemented)

The models include:

- sperical
- exponential
- gaussian
- cubic
- stable
- matérn

with all of them in a nugget and no-nugget variation. All the estimator functions are written `numba` compatible,
which will be a future dependency.
At the current stage, the package does not include any kriging. This is planned for a future release.


Installation
~~~~~~~~~~~~

You can either install scikit-gstat using pip or you download the latest version from github.

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