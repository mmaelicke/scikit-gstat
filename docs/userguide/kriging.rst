=============
Interpolation
=============

Spatial interpolation
=====================

In geostatistics the procedure of spatial interpolation is 
known as *Kriging*. That goes back to the inventor of 
Kriging, a South-African mining engineer called Dave Krige. 
He published the method in 1951.
In many text books you will also find the term *prediction*, but 
be aware that Kriging is still based on the assumption 
that the variable is a random field. THerefore I prefer the 
term *estimation* and would label the Kriging method a *BLUE*,
**B**est **L**inear **U**nbiased **E**stimator.
In general terms, the objective is to estimate a variable at 
a location that was not observed using observations from 
close locations. Kriging is considered to be the **best** 
estimator, because we utilize the spatial structure 
described by a variogram to find suitable weights for 
averaging the observations at close locations.

Given a set of observation points `s` and observation 
values at these locations :math:`Z(s)`, it can already be stated
that the estimation at an unobserved location :math:`Z^{*}(s_0)` 
is a weighted mean:

.. math::

  Z^{*}(s_0) = \sum_{i=0}^N {\lamda}_i Z(s_i)
  
where :math:`N` is the size of :math:`s` and :math:`\lambda` 
is the array of weights. This is what we want to calculate 
from a fitted variogram model.

Assumed that :math:`\lambda` had already been calculated, 
estimating the prediction is pretty straightforward:

.. ipython:: python
  :supress:
  
  import numpy as np
  
.. ipython:: python
  
  Z_s = np.array([4.2, 6.1, 0.2, 0.7, 5.2])
  lam = np.array([0.1, 0.3, 0.1, 0.1, 0.4])
  
  # calculate the weighted mean
  np.sum(Z_s * lam)
  
or shorter:

.. ipython:: python
  
  Z_s.dot(lam)
