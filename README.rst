Scikit-Gstat
============

Info: scikit-gstat needs Python >= 3.5!

.. image:: https://travis-ci.org/mmaelicke/scikit-gstat.svg?branch=version-0.1.8
    :target: https://travis-ci.org/mmaelicke/scikit-gstat
    :alt: Build Status

.. image:: https://codecov.io/gh/mmaelicke/scikit-gstat/branch/version-0.1.8/graph/badge.svg
    :target: https://codecov.io/gh/mmaelicke/scikit-gstat
    :alt: Codecov

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1345585.svg
   :target: https://doi.org/10.5281/zenodo.1345585

Deprecation Warning
-------------------

This version (0.1.8) will no longer be maintained. On the dev branch, the 0.2
version of scikit-gstat is developed and will as soon as it is stable be
merged into the master branch. This merge will be indicated by a version
number equal to or higher than 0.2 on the master branch.

The new version will not be compatible with this version of scikit-gstat.
Although most of the rewritten code affects internal data structures, some
inputs and outputs will be different. This is necessary in order to improve
the performance of scikit-gstat and keep it cleaner (in my opinion.)


Description
-----------

At current state, this module offers a scipy-styled `Variogram` class for performing geostatistical analysis.
This class can be used to derive variograms. Key benefits are a number of semivariance estimators and theoretical
variogram functions. The module is planned to be hold in the manner of scikit modules and be based upon `numpy` and
`scipy` whenever possible. There is also a distance matrix extension available, with a function for calculating
n-dimensional distance matrices for the variogram.
The estimators include:

- matheron
- cressie
- dowd
- genton (still buggy)
- entropy
- bin quantiles

The models include:

- sperical
- exponential
- gaussian
- cubic
- stable
- matérn

with all of them in a nugget and no-nugget variation. All the estimator functions are written `numba` compatible,
therefore you can just download it and include the `@jit` decorator. This can speed up the calculation for bigger
data sets up to 100x. Nevertheless, this is not included in this sckit-gstat version as these functions might be
re-implemented using Cython. This is still under evaluation.

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