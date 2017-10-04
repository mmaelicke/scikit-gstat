Scikit-Gstat
============

This module offers at the current state a scipy-styled `Variogram` class for performing geostatistical analysis.
This class is can be used to derive variograms. Key benefits are a number of semivariance estimators and theoretical
variogram functions. The module is planned to be hold in the manner of scikit modules and be based upon `numpy` and
`scipy` whenever possible. There is also a distance matrix extension available, with a function for calculating
n.dimensional distance matrices for the variogram.
The estimators include:

- matheron
- cressie
- dowd
- genton (still buggy)
- entropy (not tested)

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

At the current stage, the package does not inlcude any kriging. This is planned for a future release.


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