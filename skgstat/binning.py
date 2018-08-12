import numpy as np


def even_width_lags(distances, n, maxlag):
    """Even lag edges

    Calculate the lag edges for a given amount of bins using the same lag
    step width for all bins.

    Parameters
    ----------
    distances : numpy.array
        Flat numpy array representing the upper triangle of the distance matrix.
    n: integer
        Amount of lag classes to find
    maxlag : integer, float
        Limit the last lag class to this separating distance.

    Returns
    -------
    numpy.array

    """
    # maxlags larger than the maximum separating distance will be ignored
    if maxlag is None or maxlag > np.max(distances):
        maxlag = np.max(distances)

    return np.linspace(0, maxlag, n + 1)[1:]


def uniform_count_lags(distances, n, maxlag):
    """Uniform lag counts

    Calculate the lag edges for a given amount of bins with the same amount
    of observations in each lag class. The lag step width will be variable.

    Parameters
    ----------
    distances : numpy.array
        Flat numpy array representing the upper triangle of the distance matrix.
    n: integer
        Amount of lag classes to find
    maxlag : integer, float
        Limit the last lag class to this separating distance.

    Returns
    -------
    numpy.array

    """
    # maxlags larger than the maximum separating distance will be ignored
    if maxlag is None or maxlag > np.max(distances):
        maxlag = np.max(distances)

    # filter for distances < maxlag
    d = distances[np.where(distances <= maxlag)]

    return np.fromiter(
        (np.percentile(d, (i / n) * 100) for i in range(1, n + 1)), dtype=float
    )


def direction(X):
    """

    :param X:
    :return:
    """
    # distance
    d = lambda p1,p2: np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    # direction-angle clockwise with Y-Axis set to 0 deg.

    def d_(p1, p2):
        x = (p2[0] - p1[0], p2[1] - p1[1])
        rad = np.arctan2(*x[::-1])
        a = (-rad * 360 / (2*np.pi)) + 90
        a += 360 if a < 0 else 0
        return a

    # this is probably not the fastest way to compute this:
    m = np.matrix(np.ones((len(X), len(X))) * np.NaN)

    for i in range(len(X)):
        for j in range(len(X)):
            if i == j:
                continue
            m[i, j] = d_(X[i], X[j])

    # return
    return m


def _in_bounds(alpha, lo, up):
    """
    Returns True, if alpha is within bounds.

    :param alpha:
    :param lo:
    :param: up:
    :return: boolean if the angle is within the boounds
    """
    if lo > up:
        return alpha >= lo or alpha <= up
    else:
        return alpha >= lo and alpha <= up
