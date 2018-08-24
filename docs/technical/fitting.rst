===========================
Fitting a theoretical model
===========================

General
=======

The fit function of Variogram relies as of this writing on the
:func:`scipy.optimize.curve_fit` function. That function can
be used by ust passing a function and a set of x and y values and hoping for
the best. However, this will not always yield the best parameters. Especially
not for fitting a theoretical variogram function. There are a few assumptions
and simplifications, that we can state in order to utilize the function in a
more meaningful way.

Default fit
===========

The example below shows the performance of the fully unconstrained fit,
performed by the Levenberg-Marquardt algorithm. In scikit-gstat, this can be
used by setting the `fit_method` parameter to `lm`. However, this is not
recommended.

.. ipython:: python

    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    import numpy as np
    from skgstat.models import spherical

The fit of a spherical model will be illustrated with some made-up data
representing an experimental variogram:

.. ipython:: python

    y = [1,7,9,6,14,10,13,9,11,12,14,12,15,13]
    x = list(range(len(y)))
    xi = np.linspace(0, len(y), 100)

As the :func:`spherical <skgstat.models.spherical>` function is compiled
using numba, we wrap the function in order to let
:func:`curve_fit <scipy.optimize.curve_fit>` correctly infer the
parameters.
Then, fitting is a straightforward task.

.. ipython:: python

    def f(h, a, b):
        return spherical(h, a, b)

    cof_u, cov = curve_fit(f, x, y)
    yi = list(map(lambda x: spherical(x, *cof_u), xi))

    plt.plot(x, y, 'rD')
    @savefig fit1.png width=7in
    plt.plot(xi, yi, '--r')

In fact this looks quite good. But Levenberg-Marquardt is an unconstrained
fitting algorithm and it could likely fail on finding a parameter set. The
:func:`fit <skgstat.Variogram.fit>` method can therefore also run a box
constrained fitting algorithm. It is the Trust Region Reflective algorithm,
that will find parameters within a given range (box). It is set by the
`fit_method='tfr'` parameter and also the default setting.

Constrained fit
===============

The constrained fitting case was chosen to be the default method in skgstat
as the region can easily be specified. Furthermore it is possible to make
a good guess on initial values.
As we fit actual variogram parameters, namely the effective range, sill,
nugget and in case of a stable or Matérn model an additional shape parameter,
we know that these parameters cannot be zero. The semi-variance is defined to
be always positive.
Thus the lower bound of the region will be zero in any case. The upper limit
can easily be inferred from the experimental variogram. There are some simple
rules, that all theoretical functions follow:

    * the sill, nugget and their sum cannot be larger than the maximum
      empirical semi-variance
    * the range cannot be larger than maxlag, or if maxlag is None the
      maximum value in the distances

The :class:`Variogram <skgstat.Variogram>` class will set the bounds to
exactly these values as default behaviour. As an initial guess, it will use
the mean value of semi-variances for the sill, the mean separating distance
as range and 0 for the nugget.
In the presented empirical variogram, difference between Levenberg-Marquardt
and Trust Region Reflective is illustrated in the example below.

.. ipython:: python

    # default plot
    plt.plot(x, y, 'rD')
    plt.plot(xi, yi, '--g', label='unconstrained')

    cof, cov = curve_fit(f, x, y, p0=[3., 14.], bounds=(0, (np.max(x), np.max(y))))
    yi = list(map(lambda x: spherical(x, *cof), xi))

    plt.plot(xi, yi, '-b', label='constrained')
    @savefig fit2.png width=7in
    plt.legend(loc='lower right')

The constrained fit, represented by the solid blue line is significantly
different from the unconstrained fit (dashed, green line). The fit is overall
better as a quick RMSE calculation shows:

.. ipython:: python

    rmse_u = np.sqrt(np.sum([(spherical(_, *cof_u) - _)**2 for _ in x]))
    rmse_c = np.sqrt(np.sum([(spherical(_, *cof) - _)**2 for _ in x]))

    print('RMSE unconstrained: %.2f' % rmse_u)
    print('RMSE constrained:   %.2f' % rmse_c)

The last note about fitting a theoretical function, is that both methods
assume all lag classes to be equally important for the fit. In the specific
case of a variogram this is not true.

Distance wieghted fit
=====================

While the standard Levenberg-Marquardt and Trust Region Reflective algorithms
are both based on the idea of least squares, they assume all obersvations to
be equally important. In the specific case of a theoretical variogram
function, this is not the case.

*to be continued*