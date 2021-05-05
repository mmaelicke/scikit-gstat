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
**B** est **L** inear **U** nbiased **E** stimator.
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
  from pprint import pprint
  np.set_printoptions(precision=3)
  
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
  
  squareform(distance_matrix)
  
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

  variances = model(distance_matrix[:5])
  assert len(variances) == 5
  
Of course we could now use the inverse of these semi-variances 
to weigh the observations, **but that would not be correct.**
Remeber, that this array `variances` is what we want the 
target weights to incorporte. Whatever the weights are, these 
variances should be respected. At the same time, the five 
points among each other also have distances and therefore variances
that should be respected. Or to put it differently. 
Take the first observation point :math:`s_1`. The associated variances 
:math:`\gamma` to the other four points need to match the one 
just calculated.

.. math::

  a_1 * \gamma(s_1, s_1) + a_2 * \gamma(s_1, s_2) + a_3 * \gamma(s_1, s_3) + a_4 * \gamma(s_1, s_4) + a_5 * \gamma(s_1, s_5) =  \gamma(s_1, s_0)

Ok. First: :math:`\gamma(s_1, s_1)` is zero because the distance is obviously zero 
and the model does not have a nugget. All other distances have already been calculated.
:math:`a_1 ... a_5` are factors. These are the weights used to satisfy all given 
semi-variances. This is what we need. Obviously, we cannot calculate 5 unknown 
variables from just one equation. Lukily we have four more observations.
Writing the above equation for :math:`s_2, s_3, s_4, s_5`.
Additionally, we will write the linear equation system in matrix form as a 
dot product of the :math:`\gamma_i` and the :math:`a_i` part.

.. math::

    \begin{pmatrix}
    \gamma(s_1, s_1) & \gamma(s_1, s_2) & \gamma(s_1, s_3) & \gamma(s_1, s_4) & \gamma(s_1, s_5) \\
    \gamma(s_2, s_1) & \gamma(s_2, s_2) & \gamma(s_2, s_3) & \gamma(s_2, s_4) & \gamma(s_2, s_5) \\
    \gamma(s_3, s_1) & \gamma(s_3, s_2) & \gamma(s_3, s_3) & \gamma(s_3, s_4) & \gamma(s_3, s_5) \\
    \gamma(s_4, s_1) & \gamma(s_4, s_2) & \gamma(s_4, s_3) & \gamma(s_4, s_4) & \gamma(s_4, s_5) \\
    \gamma(s_5, s_1) & \gamma(s_5, s_2) & \gamma(s_5, s_3) & \gamma(s_5, s_4) & \gamma(s_5, s_5) \\
    \end{pmatrix} * 
    \begin{bmatrix}
    a_1 \\
    a_2 \\
    a_3 \\
    a_4 \\
    a_5\\
    \end{bmatrix} = 
    \begin{pmatrix}
    \gamma(s_0, s_1) \\
    \gamma(s_0, s_2) \\
    \gamma(s_0, s_3) \\
    \gamma(s_0, s_4) \\
    \gamma(s_0, s_5) \\
    \end{pmatrix}

That might look a bit complicated at first, but we have calculated almost everything. 
The last matrix are the `variances` that we calculated in the last step.
The first matrix is of same shape as the sqaureform distance matrix calculated in 
the very begining. All we need to do is to map the variogram model on it and 
solve the system for the matrix of factors :math:`a_1 \ldots a_5`.
In Python, there are several strategies how you could solve this problem.
Let's at first build the matrix. We need a distance matrix without 
:math:`s_0` for that.

.. ipython:: python

    dists = pdist(list(zip(x,y)))
    M = squareform(model(dists))

    pprint(M)
    pprint(variances)

And solve it:

.. ipython:: python
    :okwarning:

    from scipy.linalg import solve

    # solve for a
    a = solve(M, variances)
    pprint(a)

    # calculate estimation
    Z_s.dot(a)

That's it. Well, not really. We might have used the 
variogram and the spatial structure infered from the 
data for getting better results, but in fact our 
result is not **unbiased**. That means, the solver 
can choose any combination that satisfies the equation,
even setting everything to zero except one weight. 
That means :math:`a` could be biased.
That would not be helpful.

.. ipython:: python

    np.sum(a)

Kriging equation system
=======================

In the last section we came pretty close to the 
Kriging algorithm. The only thing missing is to 
assure unbiasedness.
The weights sum up to almost one, but they are not one.
We want to ensure, that they are always one. This 
is done by adding one more equation to the linear 
equation system. Also, we will rename the :math:`a`
array to :math:`\lambda`, which is more frequently 
used for Kriging weights. The missing equation is:

.. math::

    \sum_{i=1}^N \lambda = 1

In matrix form this changes :math:`M` to:

.. math::

    \begin{pmatrix}
    \gamma(s_1, s_1) & \gamma(s_1, s_2) & \gamma(s_1, s_3) & \gamma(s_1, s_4) & \gamma(s_1, s_5) & 1\\
    \gamma(s_2, s_1) & \gamma(s_2, s_2) & \gamma(s_2, s_3) & \gamma(s_2, s_4) & \gamma(s_2, s_5) & 1\\
    \gamma(s_3, s_1) & \gamma(s_3, s_2) & \gamma(s_3, s_3) & \gamma(s_3, s_4) & \gamma(s_3, s_5) & 1\\
    \gamma(s_4, s_1) & \gamma(s_4, s_2) & \gamma(s_4, s_3) & \gamma(s_4, s_4) & \gamma(s_4, s_5) & 1\\
    \gamma(s_5, s_1) & \gamma(s_5, s_2) & \gamma(s_5, s_3) & \gamma(s_5, s_4) & \gamma(s_5, s_5) & 1\\
    1 & 1 & 1 & 1 & 1 & 0 \\
    \end{pmatrix} * 
    \begin{bmatrix}
    \lambda_1 \\
    \lambda_2 \\
    \lambda_3 \\
    \lambda_4 \\
    \lambda_5 \\
    \mu \\
    \end{bmatrix} = 
    \begin{pmatrix}
    \gamma(s_0, s_1) \\
    \gamma(s_0, s_2) \\
    \gamma(s_0, s_3) \\
    \gamma(s_0, s_4) \\
    \gamma(s_0, s_5) \\
    1 \\
    \end{pmatrix}

This is the Kriging equation for Ordinary Kriging that can be found 
in text books. We added the ones to the result array and into the 
matrix of semivariances. :math:`\mu` is a Lagrangian multiplier 
that will be used to estimate the Kriging variance, which will 
be covered later.
Ordinary Kriging still assumes the observation and their residuals 
to be normally distributed and second order stationarity.

.. todo:: 
    Include the references to Kitanidis and Bardossy.

Applied in Python, this can be done like:

.. ipython:: python

    B = np.concatenate((variances, [1]))

    M = np.concatenate((M, [[1, 1, 1, 1, 1]]), axis=0)
    M = np.concatenate((M, [[1], [1], [1], [1], [1], [0]]), axis=1)

    weights = solve(M, B)

    # see the weights
    print('Old weights:', a)
    print('New weights:', weights[:-1])

    print('Old estimation:', Z_s.dot(a))
    print('New estimation:', Z_s.dot(weights[:-1]))
    print('Mean:', np.mean(Z_s))

And the sum of weights:

.. ipython:: python

    np.sum(weights[:-1])

The estimation did not change a lot, but the weights
perfectly sum up to one now.

Kriging error
=============

In the last step, we introduced a factor :math:`\mu`. 
It was needed to solve the linear equation system 
while assuring that the weights sum up to one. 
This factor can in turn be added to the weighted
target semi-variances used to build the equation system, 
to obtain the Kriging error.

.. ipython:: python

  sum(B[:-1] * weights[:-1]) + weights[-1]

This is really usefull when a whole map is interpolated.
Using Kriging, you can also produce a map showing
in which regions the interpolation is more certain.

Example
=======

We can use the data shown in the variography section, 
to finally interpolate the field and check the 
Kriging error. You could either build a loop around the 
code shown in the previous section, or just use 
skgstat.

.. ipython:: python
  :suppress:

  import pandas as pd 
  from skgstat import Variogram
  import matplotlib.pyplot as plt

.. ipython:: python
  :okwarning:

  data = pd.read_csv('data/sample_lr.csv')
  V = Variogram(data[['x', 'y']].values, data.z.values, 
    maxlag=90, n_lags=25, model='gaussian', normalize=False)
  
  @savefig kriging_used_variogram.png width=8in
  V.plot()

  from skgstat import OrdinaryKriging

  ok = OrdinaryKriging(V, min_points=5, max_points=20, mode='exact')

The :class:`OrdinaryKriging <skgstat.OrdinaryKriging>` class
need at least a fitted :class:`Variogram <skgstat.Variogram>` 
instance. Using `min_points` we can demand the Kriging equation 
system to be build upon at least 5 points to yield robust results.
If not enough close observations are found within the effective range
of the variogram, the estimation will not be calculated and a 
`np.NaN` value is estimated.

The `max_points` parameter will set the upper bound of the 
equation system by using in this case at last the 20 nearest points.
Adding more will most likely not change the estimation, as more points
will recieve small, if not negligible, weights.
But it will increase the processing time, as each added point will 
increase the Kriging equation system dimensionality by one.

The `mode` parameter sets the method that will 
build up the equation system. There are two implemented:
`mode='exact'` and `mode='estimate'`. Estimate is much faster, but 
if not used carefully, it can lead to numerical instability quite 
quickly. In the technical notes section of this userguide, you 
will find a whole section on the two modes.

Finally, we need the unobsered locations. The observations in 
the file were drawn from a `100x100` random field.

.. ipython:: python
  :okwarning:

  xx, yy = np.mgrid[0:99:100j, 0:99:100j]

  field = ok.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
  s2 = ok.sigma.reshape(xx.shape)

.. ipython:: python
  :suppress:
  :okwarning:

  fig, axes = plt.subplots(1, 2, figsize=(8,4))
  axes[0].imshow(field, origin='lower')
  axes[0].set_title('Kriging Interpolation')
  axes[1].imshow(s2, origin='lower', vmin=np.min(s2)*1.05, vmax=np.max(s2)*.95)
  axes[1].set_title('Kriging error')

  @savefig kriging_result_and_error.png width=8in
  fig.show()

