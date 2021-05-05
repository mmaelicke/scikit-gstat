import numpy as np
from itertools import cycle

from skgstat.Kriging import OrdinaryKriging


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
    l = n if n is not None else len(variogram.coordinates)
    indices = rng.choice(len(variogram.coordinates), replace=False, size=l)

    # TODO maybe multiprocessing?
    cros_val_map = map(_interpolate, indices, cycle([variogram]))

    # if no multiprocessing - use numpy
    deviations = np.fromiter(cros_val_map, dtype=float)

    if metric.lower() == 'rmse':
        return np.sqrt(np.mean(np.power(deviations, 2)))
    elif metric.lower() == 'mse':
        return np.mean(np.power(deviations, 2))
    else:
        # MAE
        return np.sum(np.abs(deviations)) / len(deviations)