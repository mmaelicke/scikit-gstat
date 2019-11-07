=====================
Kriging estimate mode
=====================

General
=======

Generally speaking, the kriging procedure for one unobserved point (poi) can be
broken down into three different steps.

1. calculate the distance matrix between the poi and all observed locations
   to determine the in-range points and apply the minimum and maximum points
   to be used constraints.
2. build the kriging equation system by calculating the semi-variance for all
   distances left over from step 1. Formulate squareform matrix and add the
   Lagrange multipliers
3. Solve the kriging equation system, usually by matrix inversion.

Hereby, we try to optimize the step 2 for performance. The basic idea is to
estimate the semivariances instead of calculating them on each iteration.

Why not calculate?
==================

Calculating the semivariance for all elements in the kriging equation system
gives us the best solution for the interpolation problem formulated by the
respective variogram. The main point is that the distances for each
unobserved location do differ at least slightly from all other unobserved
locations in a kriging modeling application. The variogram parameters do not
change, they are static within one modeling. This is what we want to use.
The main advantage is, that the effective range is constant in this setting.
If we can now specify a precision at which we want to resolute the range, we
can pre-calculate the corresponding semivariance values. In the time-critical
iterative formulation of the kriging equation system, one would use the
pre-calculated values of the closest distance.

What about precision?
---------------------

The precision is a hyperparameter. That means it is up to the user to decide
how precise the estimation of the kriging itself can get given an estimated
kriging equation system. The main advantage is, that the range and precision
are constant values within the scope of a simulation and therefore the
expected uncertainty can be calculated and the precision can be adjusted.
This will take some effort fine-tune the kriging instance, but it can yield
results, that are only numerically different while still increasing the
calculation time one magnitude of order.

In terms of uncertainty, one can think of a variogram function, where the
given lag distance is uncertain. This deviation can be calculated as:

.. math::
    d = \frac{range}{precision}

and increasing the precision will obviously decrease the lag deviation.

Example
=======

This example should illustrate the idea behind the estimation and show how
the precision value can influence the result. An arbitrary variogram is
created and then recalculated by the OrdinaryKriging routine to illustrate
the precision.

.. ipython:: python

    import matplotlib.pyplot as plt
    from skgstat import Variogram, OrdinaryKriging
    import numpy as np

    # create some random input
    np.random.seed(42)
    c = np.random.gamma(10, 4, size=(100,2))
    np.random.seed(42)
    v = np.random.normal(10, 2, size=100)

    V = Variogram(c, v, model='gaussian', normalize=False)
    ok = OrdinaryKriging(V, mode='exact')

    # exact calculation
    x = np.linspace(0, ok.range * 1.3, 120)
    y_c = list(map(ok.gamma_model, x))

    # estimation
    ok.mode = 'estimate'
    y_e = ok._estimate_matrix(x)

    plt.plot(x, y_c, '-b', label='exact variogram')
    @savefig krig_compare.png width=7in
    plt.plot(x, y_e, '-g', label='estimated variogram')
    plt.legend(loc='lower right')


There is almost no difference between the two lines and the result that can
be expected will be very similar, as the kriging equation system will yield
very similar weights to make the prediction.

If the precision is, however, chosen to coarse, there is a difference in the
reconstructed variogram. This way, the idea behind the estimation becomes
quite obvious.

.. ipython:: python

    # make precision really small
    ok.precision = 10

    y_e2 = ok._estimate_matrix(x)

    plt.plot(x, y_c, '-b')
    @savefig krig_coarse.png width=7in
    plt.plot(x, y_e2, '-g')