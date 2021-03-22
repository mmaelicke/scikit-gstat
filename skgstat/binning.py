import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.optimize import minimize, OptimizeWarning

from skgstat.util import shannon_entropy


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
    .. versionadded:: 0.3.8

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
    # maxlags larger than maximum separating distance will be ignored
    if maxlag is None or maxlag > np.nanmax(distances):
        maxlag = np.nanmax(distances)

    # filter for distances < maxlag
    d = distances[np.where(distances <= maxlag)]

    # calculate the edges
    edges = np.histogram_bin_edges(d, bins=method_name)[1:]

    return edges, len(edges)


def kmeans(distances, n, maxlag, binning_random_state=42, **kwargs):
    """KMeans binning
    .. versionadded:: 0.3.9

    Clustering of pairwise separating distances between locations up to
    maxlag. The lag class edges are formed equidistant from each cluster
    center. Note: this does not necessarily result in equidistance lag classes.

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

    Note
    ----
    The `KMeans <sklearn.cluster.KMeans>` that is used under the hood is not
    a deterministic algorithm, as the starting cluster centroids are seeded
    randomly. This can yield slightly different results on reach run.
    Thus, for this application, the random_state on KMeans is fixed to a
    specific value. You can change the seed by passing another seed to
    `Variogram <skgstat.Variogram>` as `binning_random_state`. 

    """
    # maxlags larger than maximum separating distance will be ignored
    if maxlag is None or maxlag > np.nanmax(distances):
        maxlag = np.nanmax(distances)

    # filter for distances < maxlag
    d = distances[np.where(distances <= maxlag)]

    # cluster the filtered distances
    km = KMeans(n_clusters=n, random_state=binning_random_state).fit(d.reshape(-1, 1))

    # get the centers
    _centers = np.sort(km.cluster_centers_.flatten())

    # build the upper edges
    bounds = zip([0] + list(_centers)[:-1], _centers)
    edges = np.fromiter(((low + up) / 2 for low, up in bounds), dtype=float)

    return edges, None


def ward(distances, n, maxlag, **kwargs):
    """Agglomerative binning
    .. versionadded:: 0.3.9

    Clustering of pairwise separating distances between locations up to
    maxlag. The lag class edges are formed equidistant from each cluster
    center. Note: this does not necessarily result in equidistance lag classes.

    The clustering is done by merging pairs of clusters that minimize the
    variance for the merged clusters, unitl `n` clusters are found.

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
    # maxlags larger than maximum separating distance will be ignored
    if maxlag is None or maxlag > np.nanmax(distances):
        maxlag = np.nanmax(distances)

    # filter for distances < maxlag
    d = distances[np.where(distances <= maxlag)]

    # cluster the filtered distances
    w = AgglomerativeClustering(linkage='ward', n_clusters=n).fit(d.reshape(-1, 1))

    # get the aggregation function
    if kwargs.get('binning_agg_func', False) == 'median':
        agg = np.median
    else:
        agg = np.mean

    # get the centers
    _centers = np.sort([agg(d[np.where(w.labels_ == i)[0]]) for i in np.unique(w.labels_)])

    # build the upper edges
    bounds = zip([0] + list(_centers)[:-1], _centers)
    edges = np.fromiter(((low + up) / 2 for low, up in bounds), dtype=float)

    return edges, None


def stable_entropy_lags(distances, n, maxlag, **kwargs):
    """Stable lags
    .. versionadded: 0.4.0

    Optimizes the lag class edges for `n` lag classes.
    The algorithm minimizes the difference between Shannon
    Entropy for each lag class. Consequently, the final
    lag classes should be of comparable uncertainty.

    Parameters
    ----------
    distances : numpy.array
        Flat numpy array representing the upper triangle of
        the distance matrix.
    n : integer
        Amount of lag classes to find
    maxlag : integer, float
        Limit the last lag class to this separating distance.

    Keyword Arguments
    -----------------
    binning_maxiter : int
        Maximum iterations before the optimization is stopped,
        if the lag edges do not converge.
    binning_entropy_bins : int, str
        Binning method for calculating the shannon entropy
        on each iteration.
    Returns
    -------
    bin_edges : numpy.ndarray
        The **upper** bin edges of the lag classes

    """
    # maxlags larger than maximum separating distance will be ignored
    if maxlag is None or maxlag > np.nanmax(distances):
        maxlag = np.nanmax(distances)

    # filter for distances < maxlag
    d = distances[np.where(distances <= maxlag)]

    # create a global binning and initial guess
    bins = np.histogram_bin_edges(d, bins=kwargs.get('binning_entropy_bins', 'sqrt'))
    initial_guess = np.linspace(0, np.nanmax(d), n + 1)[1:]

    # define the loss function
    def loss(edges):
        # get the shannon entropy for the current binning
        h = np.ones(len(edges) - 1) * 9999
        for i, bnd in enumerate(zip(edges, edges[1:])):
            l, u = bnd
            x = d[np.where((d >= l) & (d < u))[0]]
            if len(x) == 0:
                continue
            else:
                h[i] = shannon_entropy(x, bins)

        # return the absolute differences between the bins
        return np.sum(np.abs(np.diff(h)))

    # minimize the loss function
    opt = dict(maxiter=kwargs.get('binning_maxiter', 5000)) 
    res = minimize(loss, initial_guess, method='Nelder-Mead', options=opt)

    if res.success:
        return res.x, None
    else:  # pragma: no cover
        raise OptimizeWarning("Failed to find optimal lag classes.")
