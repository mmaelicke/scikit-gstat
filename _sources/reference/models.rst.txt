================
Variogram models
================

Scikit-GStat implements different theoretical variogram functions. These
model functions expect a single lag value or an array of lag values as input
data. Each function has at least a parameter `a` for the effective range and
a parameter `c0` for the sill. The nugget parameter `b` is optinal and will
be set to :math:`b:=0` if not given.

Spherical model
~~~~~~~~~~~~~~~

.. autofunction:: skgstat.models.spherical


Exponential model
~~~~~~~~~~~~~~~~~

.. autofunction:: skgstat.models.exponential


Gaussian model
~~~~~~~~~~~~~~

.. autofunction:: skgstat.models.gaussian

Cubic model
~~~~~~~~~~~

.. autofunction:: skgstat.models.cubic


Stable model
~~~~~~~~~~~~

.. autofunction:: skgstat.models.stable


Matérn model
~~~~~~~~~~~~

.. autofunction:: skgstat.models.matern