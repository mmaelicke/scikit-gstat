"""
Cross-variogram utility function. This module can be used to calcualte
cross-variograms for more than two variables, by creating a variogram
for each combination of variables.

"""
from typing import List

import numpy as np

from skgstat.Variogram import Variogram
from skgstat.DirectionalVariogram import DirectionalVariogram

def cross_variograms(coordinates: np.ndarray, values: np.ndarray, **kwargs) -> List[List[Variogram]]:
    """
    Cross-variogram matrix calculation. Similar to a cross-correlation table.
    For all combinations of ``(n_samples, N)`` given values a
    :class:`Variogram <skgstat.Variogram>` is calculated using the cross-variogram
    option between two columns into a ``(N, N)`` matrix.
    The diagonal of the *'matrix'* holds primary variograms (without cross option)
    for the respective column.
    The function accepts all keyword arguments that are also accepted by
    :class:`Variogram <skgstat.Variogram>` and 
    :class:`DirectionalVariogram <skgstat.DirectionalVariogram>` and passes them
    down to the respective function. The directional variogram will be used as
    base class if any of the specific arguments are present: azimuth, bandwidth
    or tolerance.

    Parameters
    ----------
    coordinates : numpy.ndarray, MetricSpace
        Array of shape (m, n). Will be used as m observation points of
        n-dimensions. This variogram can be calculated on 1 - n
        dimensional coordinates. In case a 1-dimensional array is passed,
        a second array of same length containing only zeros will be
        stacked to the passed one.
        For very large datasets, you can set maxlag to only calculate
        distances within the maximum lag in a sparse matrix.
        Alternatively you can supply a MetricSpace (optionally with a
        `max_dist` set for the same effect). This is useful if you're
        creating many different variograms for different measured
        parameters that are all measured at the same set of coordinates,
        as distances will only be calculated once, instead of once per
        variogram.
    values : numpy.ndarray
        Array of values observed at the given coordinates. The length of
        the values array has to match the m dimension of the coordinates
        array. Will be used to calculate the dependent variable of the
        variogram.
        If the values are of shape ``(n_samples, 2)``, a cross-variogram
        will be calculated. This assumes the main variable and the
        co-variable to be co-located under Markov-model 1 assumptions,
        meaning the variable need to be conditionally independent.

    """
    # turn input data to numpy arrays
    coordinates = np.asarray(coordinates)
    values = np.asarray(values)

    # check which base-class is needed
    if any([arg in kwargs for arg in ('azimuth', 'tolerance', 'bandwidth')]):
        BaseCls = DirectionalVariogram
    else:
        BaseCls = Variogram

    # create the output matrix
    cross_m = []

    # get the number of variables
    N = values.shape[1]

    for i in range(N):
        # create a new row
        cross_row = []
        for j in range(N):
            # check if this is a primary variogram
            if i == j:
                cross_row.append(BaseCls(coordinates, values[:, i], **kwargs))
            else:
                # extract the two datasets
                v = values[:, [i, j]]

                # append the cross-variogram
                cross_row.append(BaseCls(coordinates, v, **kwargs))

        # append
        cross_m.append(cross_row)

    return cross_m
