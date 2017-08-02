"""
Common distance functions for calculating the distance between two geometries, or a distance matrix for a set
of geometries are collected here. Most functions wrap either scipy.spatial.distance or shapely funcitonality
these functions accept numpy.matrix/numpy.ndarray objects and can be imported like:

import scikit-gstat                          # then scikit-gstat.func-name
from scikit-gstat import skgstat             # then skgstat.func-name

"""
import numpy as np
from scipy.spatial.distance import pdist as scipy_pdist, squareform


def point_dist(X, metric='euclidean', **kwargs):
    """
    Wrapper for scipy.spatial.distance.pdist function.

    Returns a distance matrix in squareform for a 1D array of x, y coordinates

    :param X: 1D array of x, y coordinates
    :param metric: will be passed to scipy_pdist
            As for now, only euclidean distances are implemented. Others will follow.
    :param kwargs: will be passes to scipy_pdist
    :return:
    """
    # check X
    _X = list(X)

    # check that all elements in the index have exactly a x and y coordinate
    if any([not len(e) == 2 for e in _X]):
        raise ValueError('The passed point data does not have a x and y coordinate for each point.')

    # data seems to be ok, return the distance matrix in squareform
    return np.matrix(squareform(scipy_pdist(_X, metric=metric, **kwargs)))


def nd_dist(X, metric='euclidean'):
    """
    Wrapper for the two euclidean matrix functions.

    The wrapper checks the dimensionality and chooses the correct matrix function.
    As for now, only euclidean distances are implemented. Others will follow.

    :param X:
    :return:
    """

    _X = np.array(X)

    # switch metric
    if metric.lower() == 'euclidean':
        # check dimensionality of elements
        if all([len(e) == 2 for e in _X]):
            return np.matrix(squareform(_euclidean_dist2D(_X)))
        # check if all coordinates have the same dimension and the dimension is not 0 or 1
        elif len(set([len(e) for e in _X])) and any(_ not in set([len(e) for e in _X]) for _ in (0, 1)):
            # N-Dimensional
            return np.matrix(squareform(_euclidean_distND(_X)))
        else:
            raise ValueError("One or more Coordinates are missing.\nPlease provide the coordinates for all values ")

    # this metric is not known
    else:
        raise ValueError("The metric '%s' is not known. Use one of: ['euclidean']" % str(metric))


def _euclidean_dist2D(X):
    """
    Returns the upper triangle of the distance matrice for an array of 2D coordinates.

    :param X: np.ndarray
    :return: upper triangle of the distance matrice
    """
    n = len(X)                  # number of pairs
    N = int((n**2 - n) / 2)     # dimension (n*n) - determinante ; half of it for upper triangle

    # define the return array
    out = np.empty(N, dtype=np.float)
    lastindex = 0               # indexing through out

    for i in np.arange(n):
        for j in np.arange(i + 1, n):       # skip determinante (j=i), use only upper (j > i)
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


def _euclidean_distND(X):
    """
    Returns the upper triangle of the distance matrix for an array of N-dimensional coordinates.
    """

    n = len(X)                  # number of pairs
    N = int((n**2 - n) / 2)     # dimension (n*n) - determinante ; half of it for upper triangle

    # define the return array
    out = np.empty(N, dtype=np.float)
    lastindex = 0               # indexing through out

    for i in np.arange(n):
        for j in np.arange(i + 1, n):       # skip determinante (j=i), use only upper (j > i)
            out[lastindex] = _d(X[i], X[j])
            lastindex += 1
    return out

