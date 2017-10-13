"""
Common distance functions for calculating the distance between two geometries, or a distance matrix for a set
of geometries are collected here. Most functions wrap either scipy.spatial.distance or shapely funcitonality
these functions accept numpy.matrix/numpy.ndarray objects and can be imported like:

import skgstat                               # then skgstat.func-name
from scikit-gstat import skgstat             # then skgstat.func-name

"""
import numpy as np
from scipy.spatial.distance import pdist as scipy_pdist, squareform
# if numba is installed uncomment
# from numba import jit
from scipy.stats import rankdata


def point_dist(X, metric='euclidean', **kwargs):
    """
    Wrapper for scipy.spatial.distance.pdist function.
    Returns a distance matrix in squareform for a 1D array of x, y coordinates

    :param X: 1D array of x, y coordinates
    :param metric: will be passed to scipy_pdist
            As for now, only euclidean distances are implemented. Others will follow.
    :param kwargs: will be passes to scipy_pdist
    :return: distance matrix of the given coordinates
    """
    # check X
    _X = list(X)

    # switch metric
    if metric.lower() == 'euclidean':
        # check that all elements in the index have exactly a x and y coordinate
        if any([not len(e) == 2 for e in _X]):
            raise ValueError('The passed point data does not have a x and y coordinate for each point.')

        # data seems to be ok, return the distance matrix in squareform
        return np.matrix(squareform(scipy_pdist(_X, metric=metric, **kwargs)))

    elif metric.lower() == 'rank':
        return np.matrix(rankdata(point_dist(X, metric='euclidean')))

    # this metric is not known
    else:
        raise ValueError("The metric '%s' is not known. Use one of: ['euclidean', 'rank']" % str(metric))


def nd_dist(X, metric='euclidean'):
    """
    Wrapper for the different distance functions.

    The wrapper checks the dimensionality and chooses the correct matrix function.
    In case the metric is 'rank', the result of metric='euclidean' (either 2 or N dimensional) will be ranked using
    scipy.stats.rankdata function.

    :param X: 1D array of x, y coordinates
    :return: distance matrix of the given coordinates
    """

    _X = np.array(X)

    # switch metric
    if metric.lower() == 'euclidean':
        # check dimensionality of elements
        if all([len(e) == 2 for e in _X]):
            return np.matrix(squareform(_euclidean_dist2D(_X)))
        # check if all coordinates have the same dimension and the array is not empty
        elif len(set([len(e) for e in _X])) == 1 and len(_X[0]) != 0:
            # N-Dimensional
            return np.matrix(squareform(_euclidean_distND(_X)))
        else:
            raise ValueError("One or more Coordinates are missing.\nPlease provide the coordinates for all values ")

    elif metric.lower() == 'rank':
        return np.matrix(rankdata(nd_dist(X, metric='euclidean')))

    # this metric is not known
    else:
        raise ValueError("The metric '%s' is not known. Use one of: ['euclidean', 'rank']" % str(metric))


# if numba is installed uncommment
#@jit
def _euclidean_dist2D(X):
    """
    Returns the upper triangle of the distance matrix for an array of 2D coordinates.

    :param X: np.ndarray of the x, y coordinates
    :return: upper triangle of the distance matrice
    :param X: np.ndarray
    :return: upper triangle of the distance matrix
    """
    n = len(X)                  # number of pairs
    N = int((n**2 - n) / 2)     # dimension (n*n) - diagonal; half of it for upper triangle

    # define the return array
    out = np.empty(N, dtype=np.float)
    lastindex = 0               # indexing through out

    for i in np.arange(n):
        for j in np.arange(i + 1, n):       # skip diagonal (j=i), use only upper (j > i)
            out[lastindex] = np.sqrt( (X[i][0] - X[j][0])**2 + (X[i][1] - X[j][1])**2 )
            lastindex += 1
    return out


#_d = lambda p1, p2: np.sqrt(np.sum([np.diff(tup)**2 for tup in zip(p1, p2)]))
def _d(p1, p2):
    s = 0.0
    for i in range(len(p1)):
        s += (p1[i] - p2[i])**2
    return np.sqrt(s)

pyd = lambda p1, p2: np.sqrt(np.sum([np.diff(tup)**2 for tup in zip(p1, p2)]))


# if numba is installed uncommment
#@jit
def _euclidean_distND(X):
    """
    Returns the upper triangle of the distance matrix for an array of N-dimensional coordinates.
    :param X: np.array of n-dimensional coordinates
    :return: upper triangle of the distance matrix
    """

    n = len(X)                  # number of pairs
    N = int((n**2 - n) / 2)     # dimension (n*n) - diagonal ; half of it for upper triangle

    # define the return array
    out = np.empty(N, dtype=np.float)
    lastindex = 0               # indexing through out

    for i in np.arange(n):
        for j in np.arange(i + 1, n):       # skip diagonal (j=i), use only upper (j > i)
            out[lastindex] = _d(X[i], X[j])
            lastindex += 1
    return out

