import numpy as np
from itertools import cycle

from skgstat.Kriging import OrdinaryKriging
from skgstat.Variogram import Variogram
from skgstat.util.likelihood import get_likelihood


def _interpolate(idx: int, variogram) -> float:
    # get the data for this iteration
    c = np.delete(variogram.coordinates, idx, axis=0)
    v = np.delete(variogram.values, idx, axis=0)
    ok = OrdinaryKriging(variogram, coordinates=c, values=v)

    # interpolate Z[idx]
    Z = ok.transform(
        [variogram.coordinates[idx][0]],
        [variogram.coordinates[idx][1]]
    )

    return (Z - variogram.values[idx])[0]


def jacknife(
    variogram,
    n: int = None,
    metric: str = 'rmse',
    seed=None
) -> float:
    """
    Leave-one-out cross validation of the given variogram
    model using the OrdinaryKriging instance.
    This method can be called using
    :func:`Variogram.cross_validate <skgstat.Variogram.cross_validate>`.

    Parameters
    ----------
    variogram : skgstat.Variogram
        The variogram isnstance to be validated
    n : int
        Number of points that should be used for cross validation.
        If None is given, all points are used (default).
    metric : str
        Metric used for cross validation. Can be one of
        ['rmse', 'mse', 'mae']

    Returns
    -------
    metric : float
        Cross-validation result The value is given
        in the selected metric.

    """
    if metric.lower() not in ('rmse', 'mse', 'mae'):
        raise ValueError("metric has to be in ['rmse', 'mse', 'mae']")

    # shuffle the input coordinates
    rng = np.random.default_rng(seed=seed)
    size = n if n is not None else len(variogram.coordinates)
    indices = rng.choice(len(variogram.coordinates), replace=False, size=size)

    # TODO maybe multiprocessing?
    cros_val_map = map(_interpolate, indices, cycle([variogram]))

    # if no multiprocessing - use numpy
    deviations = np.fromiter(cros_val_map, dtype=float)

    if metric.lower() == 'rmse':
        return np.sqrt(np.nanmean(np.power(deviations, 2)))
    elif metric.lower() == 'mse':
        return np.nanmean(np.power(deviations, 2))
    else:
        # MAE
        return np.nansum(np.abs(deviations)) / len(deviations)


def aic(variogram: Variogram) -> float:
    like = get_likelihood(variogram)

    # get parameters
    params = variogram.parameters
    k = len(params)
    if params[-1] < 1e-6:
        k -= 1

    # get maximum log-likelihood
    log_like = like(params)

    # return AIC
    return 2 * k - 2 * log_like


def bic(variogram: Variogram) -> float:
    like = get_likelihood(variogram)

    # get parameters
    params = variogram.parameters
    k = len(params)
    if params[-1] < 1e-6:
        k -= 1

    # get maximum log-likelihood
    log_like = like(params)

    # return BIC
    return 2 * np.log(k) - 2 * log_like
