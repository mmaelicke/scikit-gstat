"""
This module implements a maximum likelihood function for variogram models.
The definition is taken from [601]_:

References
----------
[601]   Lark, R. M. "Estimating variograms of soil properties by the 
        method‐of‐moments and maximum likelihood." European Journal 
        of Soil Science 51.4 (2000): 717-728.
"""
from typing import Callable, List
from itertools import cycle

import numpy as np
from scipy.spatial.distance import squareform
from scipy.linalg import inv, det

from skgstat import Variogram


DOC_TEMPLATE = """Autocorrelation function.
This function calcualtes the sptial autocorrelation for any
model function only, by setting nugget to 0 and sill to 1.
This can be used to create an autocorreation matrix as used
to derive a maximum likelihhod function for the model.

Original documentation:
{doc}
"""


def _model_transformed(model_func, has_s: bool = False) -> Callable:
    """
    Transforms the model parameter input to fit likelihood function
    input parameters.
    The returned function can be used to create the log-likelihood
    function that has to be minimized.
    To build up the spatial autocorrelation matrix, the spatial
    autocorrelation has to be separated from nugget and sill.
    """
    if has_s:
        def wrapped(h, r, s):
            return model_func(h, r, s=s, c0=1, b=0)
    else:
        def wrapped(h, r):
            return model_func(h, r, c0=1, b=0)

    # add the original docstring
    wrapped.__doc__ = DOC_TEMPLATE.format(doc=model_func.__doc__)
    wrapped.__name__ = f"autocorr_{model_func.__name__}"

    return wrapped


def _build_A(transformed_func: Callable, params: List[float], dists: np.ndarray) -> np.ndarray:
    """
    Builds the autocorrelation matrix for a given model function.
    """
    if len(params) == 4:
        r, c0, s, b = params
        a = np.fromiter(map(transformed_func, dists, cycle([r]), cycle([s])), dtype=float)
    else:
        r, c0, b = params
        # calcualte the upper triangle of A:
        a = np.fromiter(map(transformed_func, dists, cycle([r])), dtype=float)

    # build the full matrix
    A = squareform((c0 / (c0 + b)) * (1 - a))

    # replace diagonal 0 with ones
    np.fill_diagonal(A, 1)

    return A


def get_likelihood(variogram: Variogram) -> Callable:
    """
    """
    # extract the current data
    values = variogram.values
    # TODO: there is a bug as this is not working:
    # dists = variogram.distance
    try:
        dists = squareform(variogram._X.dists.todense())
    except AttributeError:
        dists = variogram.distance

    # get the transformed model func
    has_s = variogram.model.__name__ in ('matern', 'stable')
    transformed_func = _model_transformed(variogram.model, has_s=has_s)

    def likelihood(params: List[float]) -> float:
        # calculate A
        A = _build_A(transformed_func, params, dists)
        n = len(A)

        # invert A
        A_inv = inv(A)

        # build the 1 vector and t
        ones = np.ones((n, 1))
        z = values.reshape(n, -1)

        # build the estimate matrix for field means
        m = inv(ones.T @ A_inv @ ones) @ (ones.T @ A_inv @ z)
        b = np.log((z - m).T @ A_inv @ (z - m))

        # get the log of determinant of A
        D = np.log(det(A))
        # np.log(0.0) is -inf, so we need to check for this
        if D == -np.inf:
            # print a warning and return np.inf to not use these parameters
            print("Warning: D = -inf, returning np.inf")
            return np.inf

        # finally log-likelihood of the model given parameters
        loglike = (n / 2)*np.log(2*np.pi) + (n / 2) - (n / 2) * np.log(n) + 0.5 * D + (n / 2) * b

        # this is actually a 1x1 matrix
        return loglike.flatten()[0]

    # return the likelikhood function
    return likelihood
