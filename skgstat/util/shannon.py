import numpy as np


def shannon_entropy(x, bins):
    """Shannon Entropy

    Calculates the Shannon Entropy, which is the most basic
    metric in information theory. It can be used to calculate
    the information content of discrete distributions.
    This can be used to estimate the intrinsic uncertainty of
    a sample,independent of the value range or variance, which
    makes it more comparable.

    Parameters
    ----------
    x : numpy.ndarray
        flat 1D array of the observations
    bins : list, int
        upper edges of the bins used to calculate the histogram
        of x.
    
    Returns
    -------
    h : float
        Shannon Entropy of x, given bins.
    """
    # histogram
    c, _ = np.histogram(x, bins=bins)

    # empirical probabilities
    p = c / np.sum(c) + 1e-15

    # map information function and return product
    return - np.fromiter(map(np.log2, p), dtype=float).dot(p)
