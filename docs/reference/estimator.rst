===================
Estimator Functions
===================

Scikit-GStat implements various semi-variance estimators. These fucntions can
be found in the skgstat.estimators submodule. Each of these functions can be
used independently from Variogram class. In this case the estimator is
expecting an array of pairwise differences to calculate the semi-variance.
Not the values themselves.

Matheron
~~~~~~~~

.. autofunction:: skgstat.estimators.matheron

Cressie
~~~~~~~

.. autofunction:: skgstat.estimators.cressie

Dowd
~~~~

.. autofunction:: skgstat.estimators.dowd

Genton
~~~~~~

.. autofunction:: skgstat.estimators.genton

Shannon Entropy
~~~~~~~~~~~~~~~

.. autofunction:: skgstat.estimators.entropy


MinMax
~~~~~~

.. warning::

    This is an experimental semi-variance estimator. It is heavily influenced
    by extreme values and outliers. That behaviour is usually not desired in
    geostatistics.

.. autofunction:: skgstat.estimators.minmax


Percentile
~~~~~~~~~~

.. warning::

    This is an experimental semi-variance estimator. It uses just a
    percentile of the given pairwise differences and does not bear any
    information about their variance.

.. autofunction:: skgstat.estimators.percentile