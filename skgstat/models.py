import numpy as np
import math
from scipy import special


class Variogram_Wrapper:
    """
    The Variogram Wrapper should be used to decorate the mathematical expression of the variogram function. The typical
    signature is

    func(h, *args)

    When decorated by Variogram_Wrapper, func will accept iterables as well and will be called in a list comprehension.
    The Wrapper preserves the original __name__ attribute of the function.

    """

    def __init__(self, func):
        """

        :param func:
        """
        self.func = func
        self.__name__ = func.__name__

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        if hasattr(args[0], '__iter__'):
            # this is an iterable
            new_args = args[1:]
            return np.array([self.func(value, *new_args, **kwargs) for value in args[0]])
#            return np.fromiter(map(self.func, args[0], *new_args, **kwargs), dtype=np.float)
        else:
            return self.func(*args, **kwargs)

# --- generic theoretical Variogram Models --- #


@Variogram_Wrapper
def spherical(h, a, C0, b=0):
    """
    The Spherical variogram function.

    :param h:   the separation lag
    :param a:   the range parameter (not effective range!)
    :param C0:  the Variogram sill
    :param b:   the Variogram nugget

    :return:    float, or list of; the semivariance at separation lag h
    """
    # prepare parameters
    r = a / 1.
    C0 -= b

    if h <= a:
        return b + C0 * ((1.5 * (h / r)) - (0.5 * ((h / r)**3.0)))
    else:
        return b + C0


@Variogram_Wrapper
def exponential(h, a, C0, b=0):
    """
    The Exponential variogram function.

    :param h:   the separation lag
    :param a:   the range parameter (not effective range!)
    :param C0:  the Variogram sill
    :param b:   the Variogram nugget

    :return:    float, or list of; the semivariance at separation lag h
    """
    # prepare parameters
    r = a / 3.
    C0 -= 0

    try:
        return b + C0 * (1. - math.exp(-(h / r)))
    except:
        return b + C0

@Variogram_Wrapper
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
    r = a / np.sqrt(3)
    C0 -= b

    return b + C0 * (1. - math.exp(- (h ** 2 / r ** 2)))


@Variogram_Wrapper
def cubic(h, a, C0, b=0):
    """
    The Cubic Variogram function

    :param h:   the separation lag
    :param a:   the range parameter (not effective range!)
    :param C0:  the Variogram sill
    :param b:   the Variogram nugget
    """
    # prepare parameters
    C0 -= b

    if h <= a:
        return b + C0 * ( (7*(h**2 / a**2)) - ((35/4)*(h**3/a**3)) + ((7/2)*(h**5/a**5)) - ((3/4)*(h**7/a**7)) )
    else:
        return b + C0


@Variogram_Wrapper
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
    C0 -= b

    if s > 2:
        s = 2
    return b + C0 * (1. - math.exp(- math.pow(h / r, s)) )


@Variogram_Wrapper
def matern(h, a, C0, s, b=0):
    """
    The Matérn model.

    :param h:   lag
    :param a:   range
    :param C0:  sill
    :param s:   smoothness parameter
    :param b:   nugget
    :return:
    """
    # prepare parameters
    r = a
    C0 -= b

    return b + C0 * (1 - ( (1 / (np.power(2, s - 1) * special.gamma(s))) * np.power(h / r, s) * special.kv(s, h / r) ))


# --- Adaptions using no nugget effect --- #
def debug_spherical(h, a, C0):
    if isinstance(h, list) or isinstance(h, np.ndarray):
        return np.array([debug_spherical(_, a, C0)  for _ in h])
    else:
        if h <= a:
            return C0 * ((1.5*h/a) - (0.5*(h/a)))**3
        else:
            return C0