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

  Z^{*}(s_0) = \sum_{i=0}^N {\lambda}_i Z(s_i)
  
where :math:`N` is the size of :math:`s` and :math:`\lambda` 
is the array of weights. This is what we want to calculate 
from a fitted variogram model.

Assumed that :math:`\lambda` had already been calculated, 
estimating the prediction is pretty straightforward:

.. ipython:: python
  :suppress:
  
  import numpy as np
  from scipy.spatial.distance import pdist, squareform
  
.. ipython:: python
  
  Z_s = np.array([4.2, 6.1, 0.2, 0.7, 5.2])
  lam = np.array([0.1, 0.3, 0.1, 0.1, 0.4])
  
  # calculate the weighted mean
  np.sum(Z_s * lam)
  
or shorter:

.. ipython:: python
  
  Z_s.dot(lam)

In the example above the weights were just made up. 
Now we need to understand how this array of weights 
can be calculated.

Using a spatial model
=====================

Instead of just making up weights, we will now learn 
how we can utilize a variogram model to calculate the weights.
At its core a variogram describes how point observations become 
more dissimilar with distance. Point distances can easily be calculated, 
not only for observed locations, but also for unobserved locations.
As the variogram is only a function of *distance*, we can easily 
calculate a semi-variance value for any possible combination of point
pairs. 

Assume we have five close observations for an unobserved location, 
like in the example above. Instead of making up weights, we can use 
the semi-variance value as a weight, as a first shot. 
What we still need are locations and a variogram model. For both, 
we can just make something up.

.. ipython:: python

  x = np.array([4.0, 2.0, 4.1, 0.3, 2.0])
  y = np.array([5.5, 1.2, 3.7, 2.0, 2.5])
  z = np.array([4.2, 6.1, 0.2, 0.7, 5.2])
  
  s0 = [2., 2.]
  
  distance_matrix = pdist([s0] + list(zip(x,y)))
  
  distance_matrix
  
Next, we build up a variogram model of spherical shape, that uses a 
effective range larger than the distances in the matrix. Otherwise, 
we would just calcualte the arithmetic mean.

.. ipython:: python

  from skgstat.models import spherical
  
  # range= 7. sill = 2. nugget = 0.
  model = lambda h: spherical(h, 7.0, 2.0, 0.0)
  
The distances to the first point `s0` are the first 5 elements in 
the distance matrix. Therefore the semi-variances are calculated 
straightforward.

.. ipython:: python

  variances = model(distance_matrx[:5])
  assert len(variances) == 5
  

