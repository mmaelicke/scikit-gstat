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
    if maxlag is None or maxlag > np.nanmax(distances):
        maxlag = np.nanmax(distances)

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
    if maxlag is None or maxlag > np.nanmax(distances):
        maxlag = np.nanmax(distances)

    # filter for distances < maxlag
    d = distances[np.where(distances <= maxlag)]

    return np.fromiter(
        (np.nanpercentile(d, (i / n) * 100) for i in range(1, n + 1)),
        dtype=float
    )
