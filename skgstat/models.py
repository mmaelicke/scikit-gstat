import math

import numpy as np
from scipy import special
from numba import jit


class variogram_function:
    """
    The Variogram Wrapper should be used to decorate the mathematical
    expression of the variogram function. The typical signature is

    func(h, *args)

    When decorated by Variogram_Wrapper, func will accept iterables as well
    and will be called in a list comprehension.
    The Wrapper preserves the original __name__ and __doc__ attribute of the
    function.

    """

    def __init__(self, func):
        """

        :param func:
        """
        self.func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        if hasattr(args[0], '__iter__'):
            # this is an iterable
            new_args = args[1:]
            return np.array(
                [self.func(value, *new_args, **kwargs) for value in args[0]]
            )
        else:
            return self.func(*args, **kwargs)


@variogram_function
@jit
def spherical(h, r, c0, b=0):
    r"""Spherical Variogram function

    Implementation of the spherical variogram function. Will calculate the
    dependent variable for a given lag (h). The nugget (b) defaults to be 0.

    Parameters
    ----------
    h : float
        Specifies the lag of separating distances that the dependent variable
        shall be calculated for. It has to be a positive real number.
    r : float
        The effective range. Note this is not the range parameter! However,
        for the spherical variogram the range and effective range are the same.
    c0 : float
        The sill of the variogram, where it will flatten out. The function
        will not return a value higher than C0 + b.
    b : float
        The nugget of the variogram. This is the value of independent
        variable at the distance of zero. This is usually attributed to
        non-spatial variance.

    Returns
    -------
    gamma : numpy.float64
        Unlike in most variogram function formulas, which define the function
        for :math:`2*\gamma`, this function will return :math:`\gamma` only.

    Notes
    -----
    The implementation follows _[10]:

    .. math::
        \gamma = b + C_0 * \left({1.5*\frac{h}{r} - 0.5*{\frac{h}{r]^3}\right)

    if :math:`h < r`, and

    .. math::
        \gamma = b + C_0

    else. r is the effective range, which is in case of the spherical
    variogram just a.

    References
    ----------
    .. [10]: Burgess, T. M., & Webster, R. (1980). Optimal interpolation
       and isarithmic mapping of soil properties. I.The semi-variogram and
       punctual kriging. Journal of Soil and Science, 31(2), 315–331,
       http://doi.org/10.1111/j.1365-2389.1980.tb02084.x

    """
    # prepare parameters
    a = r / 1.

    if h <= r:
        return b + c0 * ((1.5 * (h / a)) - (0.5 * ((h / a) ** 3.0)))
    else:
        return b + c0


@variogram_function
@jit
def exponential(h, r, c0, b=0):
    r"""Exponential Variogram function

    Implementation of the spherical variogram function. Will calculate the
    dependent variable for a given lag (h). The nugget (b) defaults to be 0.

    Parameters
    ----------
    h : float
        Specifies the lag of separating distances that the dependent variable
        shall be calculated for. It has to be a positive real number.
    r : float
        The effective range. Note this is not the range parameter! For the
        exponential variogram function the range parameter a is defined to be
        :math:`a=\frac{r}{3}`. The effetive range is the lag where 95% of the
        sill are exceeded. This is needed as the sill is only approached
        asymtotically by an exponential function.
    c0 : float
        The sill of the variogram, where it will flatten out. The function
        will not return a value higher than C0 + b.
    b : float
        The nugget of the variogram. This is the value of independent
        variable at the distance of zero. This is usually attributed to
        non-spatial variance.

    Returns
    -------
    gamma : numpy.float64
        Unlike in most variogram function formulas, which define the function
        for :math:`2*\gamma`, this function will return :math:`\gamma` only.

    Notes
    -----
    The implementation following _[11] and _[12] is as:
    .. math::
        \gamma = b + C_0 * \left({1 - e^{-\frac{h}{a}}}\right)

    a is the range parameter, that can be calculated from the
    effective range r as: :math:`a = \frac{r}{3}`.

    References
    ----------

    .. [11]: Cressie, N. (1993): Statistics for spatial data.
       Wiley Interscience.

    .. [12]: Chiles, J.P., Delfiner, P. (1999). Geostatistics. Modeling Spatial
       Uncertainty. Wiley Interscience.

    """
    # prepare parameters
    a = r / 3.

    return b + c0 * (1. - math.exp(-(h / a)))


@variogram_function
# @jit
def gaussian(h, a, C0, b=0):
    """
    The Gaussian variogram function.

    :param h:   the separation lag
    :param a:   the range parameter (not effective range!)
    :param C0:  the Variogram sill
    :param b:   the Variogram nugget

    :return:    float, or list of; the semivariance at separation lag h
    """
    # prepare parameters
    #r = a / np.sqrt(3)
    r = a / 2.
    #C0 -= b

    return b + C0 * (1. - math.exp(- (h ** 2 / r ** 2)))


@variogram_function
# if numba is installed uncommment
# @jit
def cubic(h, a, C0, b=0):
    """
    The Cubic Variogram function

    :param h:   the separation lag
    :param a:   the range parameter (not effective range!)
    :param C0:  the Variogram sill
    :param b:   the Variogram nugget
    """
    # prepare parameters
    #C0 -= b

    if h <= a:
        return b + C0 * ( (7*(h**2 / a**2)) - ((35/4)*(h**3/a**3)) + ((7/2)*(h**5/a**5)) - ((3/4)*(h**7/a**7)) )
    else:
        return b + C0


@variogram_function
@jit
def stable(h, a, C0, s, b=0):
    """
    The  Stable Variogram function.

    :param h:
    :param a:
    :param C0:
    :param s:
    :param b:
    :return:
    """
    # prepare parameters
    r = a * math.pow(3, 1 / s)
    #C0 -= b

#    if s > 2:
#        s = 2
    return b + C0 * (1. - math.exp(- math.pow(h / r, s)))


@variogram_function
# if numba is installed uncommment
# @jit
def matern(h, a, C0, s, b=0):
    """
    The Matérn model.

    For Matérn function see:
    Minasny, B., & McBratney, A. B. (2005). The Matérn function as a general model for soil variograms.
        Geoderma, 128(3–4 SPEC. ISS.), 192–207. http://doi.org/10.1016/j.geoderma.2005.04.003.

    :param h:   lag
    :param a:   range
    :param C0:  sill
    :param s:   smoothness parameter
    :param b:   nugget
    :return:
    """
    # prepare parameters
    r = a
    #C0 -= b

    return b + C0 * (1 - ( (1 / (np.power(2, s - 1) * special.gamma(s))) * np.power(h / r, s) * special.kv(s, h / r) ))
