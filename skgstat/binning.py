import numpy as np


def even_width_lags(distances, n, maxlag):
    """Even lag edges

    Calculate the lag edges for a given amount of bins using the same lag
    step width for all bins.

    .. versionchanged:: 0.3.8
        Function returns `None` as second value to indicate that
        The number of lag classes was not changed

    Parameters
    ----------
    distances : numpy.array
        Flat numpy array representing the upper triangle of
        the distance matrix.
    n : integer
        Amount of lag classes to find
    maxlag : integer, float
        Limit the last lag class to this separating distance.

    Returns
    -------
    bin_edges : numpy.ndarray
        The **upper** bin edges of the lag classes

    """
    # maxlags larger than the maximum separating distance will be ignored
    if maxlag is None or maxlag > np.nanmax(distances):
        maxlag = np.nanmax(distances)

    return np.linspace(0, maxlag, n + 1)[1:], None


def uniform_count_lags(distances, n, maxlag):
    """Uniform lag counts

    Calculate the lag edges for a given amount of bins with the same amount
    of observations in each lag class. The lag step width will be variable.

    .. versionchanged:: 0.3.8
        Function returns `None` as second value to indicate that
        The number of lag classes was not changed

    Parameters
    ----------
    distances : numpy.array
        Flat numpy array representing the upper triangle of
        the distance matrix.
    n : integer
        Amount of lag classes to find
    maxlag : integer, float
        Limit the last lag class to this separating distance.

    Returns
    -------
    bin_edges : numpy.ndarray
        The **upper** bin edges of the lag classes

    """
    # maxlags larger than the maximum separating distance will be ignored
    if maxlag is None or maxlag > np.nanmax(distances):
        maxlag = np.nanmax(distances)

    # filter for distances < maxlag
    d = distances[np.where(distances <= maxlag)]

    return np.fromiter(
        (np.nanpercentile(d, (i / n) * 100) for i in range(1, n + 1)),
        dtype=float
    ), None


def auto_derived_lags(distances, method_name, maxlag):
    """Derive bins automatically
    .. vserionadded:: 0.3.8

    Uses `histogram_bin_edges <numpy.histogram_bin_edges>` to derive the
    lag classes automatically. Supports any method supported by
    `histogram_bin_edges <numpy.histogram_bin_edges>`. It is recommended
    to use `'stuges'`, `'doane'` or `'fd'`.

    Parameters
    ----------
    distances : numpy.array
        Flat numpy array representing the upper triangle of
        the distance matrix.
    maxlag : integer, float
        Limit the last lag class to this separating distance.
    method_name : str
        Any method supported by
        `histogram_bin_edges <numpy.histogram_bin_edges>`

    Returns
    -------
    bin_edges : numpy.ndarray
        The **upper** bin edges of the lag classes

    See Also
    --------
    numpy.histogram_bin_edges

    """
    # maxlags largher than maximum separating distance will be ignored
    if maxlag is None or maxlag > np.nanmax(distances):
        maxlag = np.nanmax(distances)

    # filter for distances < maxlag
    d = distances[np.where(distances <= maxlag)]

    # calculate the edges
    edges = np.histogram_bin_edges(d, bins=method_name)[1:]

    return edges, len(edges)
