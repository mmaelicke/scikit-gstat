"""
This file collects different functions for estimating the semivariance of a group of values. These functions
can be used to fit a experimental variogram to a list of points. Each of the given functions just calculate
the estimate for one bin. If you want a Variogram, use the variogram functions or Class from the Variogram and vario
submodule, or order the bins yourself
"""
import numpy as np
from scipy.special import binom
from scipy.stats import entropy as scipy_entropy


def matheron(X, power=2):
    """
    Return the Matheron Variogram of the given sample X.
    X has to be an even-length array of point pairs like: x1, x1+h, x2, x2+h ...., xn, xn + h.
    If X.ndim > 1, matheron will be called recursively and a list of Matheron Variances is returned.

    Matheron, G. (1965): Les variables regionalisées et leur estimation. Editions Masson et Cie, 212 S., Paris.
    Matheron, G. (1962): Traité de Géostatistique Appliqué, Tonne 1. Memoires de Bureau de Recherches Géologiques et Miniéres, Paris.

    :param X:
    :param power:
    :return:
    """
    _X = np.array(X)

    if any([isinstance(_, list) or isinstance(_, np.ndarray) for _ in _X]):
        return np.array([matheron(_, power=power) for _ in _X])

    # check even
    if len(_X) % 2 > 0:
        raise ValueError('The sample does not have an even length: {}'.format(_X))

    # calculate:
    if len(_X) == 0:
        # would give ZeroDivisionError
        return np.nan
    return (1 / len(_X)) * np.nansum([np.power(_X[i] - _X[i + 1], power) for i in np.arange(0, len(_X) - 1, 2)])


def cressie(X):
    """
    Return the Cressie-Hawkins Variogram of the given sample X.
    X has to be an even-length array of point pairs like: x1, x1+h, x2, x2+h ...., xn, xn + h.
    If X.ndim > 1, cressie will be called recursively and a list of Cressie-Hawkins Variances is returned.

    Cressie, N., and D. Hawkins (1980): Robust estimation of the variogram. Math. Geol., 12, 115-125.

    :param X:
    :return:
    """
    _X = np.array(X)

    if any([isinstance(_, list) or isinstance(_, np.ndarray) for _ in _X]):
        return np.array([cressie(_) for _ in _X])

    # check even
    if len(_X) % 2 > 0:
        raise ValueError('The sample does not have an even length: {}'.format(_X))

    # calculate
    N = 0.5 * len(_X)
    if N == 0:
        # would raise ZeroDivisonError
        return np.nan

    term1 = (1 / N) * np.nansum([ np.sqrt(np.abs(_X[i] - _X[i + 1])) for i in np.arange(0, len(_X) - 1, 2)])
    term2 = 0.457 + (0.494 / N) + (0.045 / np.power(N, 2))

    return 0.5 * np.power(term1, 4) / term2


def dowd(X):
    """
    Return the Dowd Variogram of the given sample X.
    X has to be an even-length array of point pairs like: x1, x1+h, x2, x2+h ...., xn, xn + h.
    If X.ndim > 1, dowd will be called recursively and a list of Cressie-Hawkins Variances is returned.

    Dowd, P. A., (1984): The variogram and kriging: Robust and resistant estimators, in Geostatistics for Natural
        Resources Characterization. Edited by G. Verly et al., pp. 91 - 106, D. Reidel, Dordrecht.

    :param X:
    :return:
    """
    _X = np.array(X)

    if any([isinstance(_, list) or isinstance(_, np.ndarray) for _ in _X]):
        return np.array([dowd(_) for _ in _X])

    # check even
    if len(_X) % 2 > 0:
        raise ValueError('The sample does not have an even length: {}'.format(_X))

    # calculate
    term1 = np.nanmedian([np.abs(_X[i] - _X[i + 1]) for i in np.arange(0, len(_X) - 1, 2)])
    return 0.5 * (2.198 * np.power(term1, 2))


def genton(X):
    """
    Return the Genton Variogram of the given sample X.
    X has to be an even-length array of point pairs like: x1, x1+h, x2, x2+h ...., xn, xn + h.
    If X.ndim > 1, genton will be called recursively and a list of Cressie-Hawkins Variances is returned.

    Genton, M. G., (1998): Highly robust variogram estimation, Math. Geol., 30, 213 - 221.

    :param X:
    :return:
    """
    _X = np.array(X)

    if any([isinstance(_, list) or isinstance(_, np.ndarray) for _ in _X]):
        return np.array([genton(_) for _ in _X])

    # check even
    if len(_X) % 2 > 0:
        raise ValueError('The sample does not have an even length: {}'.format(_X))

    # calculate
    try:
        y = [np.abs( (_X[i] - _X[i + 1]) - (_X[j]) - _X[j + 1] ) for i in np.arange(0, len(_X), 2) for j in np.arange(0, len(_X), 2) if i < j]

        # get k  k is binom(N(x)/2+1, 2)
        k = binom(int(len(_X) / 2 + 1), 2)

        # return the kth percentile
        return 0.5 * np.power(2.219 * np.percentile(y, k), 2)
    except ZeroDivisionError:
        return np.nan


def minmax(X):
    """
    Returns the MinMax Semivariance of sample X.
    X has to be an even-length array of point pairs like: x1, x1+h, x2, x2+h, ..., xn, xn+h.

    :param X:
    :return:
    """
    _X = np.asarray(X)

    if any([isinstance(_, list) or isinstance(_, np.ndarray) for _ in _X]):
        return [minmax(_) for _ in _X]

    # check even
    if len(_X) % 2 > 0:
        raise ValueError('The sample does not have an even length: {}'.format(_X))

    return (np.nanmax(_X) - np.nanmin(_X)) / np.nanmean(_X)



def entropy(X):
    """
    Use the Shannon Entropy H to describe the distribution of the given sample

    :param X:
    :return:
    """
    _X = np.array(X)

    if any([isinstance(_, (list, np.ndarray)) for _ in _X]):
        return np.array([entropy(_) for _ in _X])

    # check even
    if len(_X) % 2 > 0:
        raise ValueError('The sample does not have an even length: {}'.format(_X))

    # calculate
    vals = [np.abs(_X[i] - _X[i + 1]) for i in np.arange(0, len(_X), 2)]

    return scipy_entropy(pk=np.histogram(vals, bins='fd')[0])
