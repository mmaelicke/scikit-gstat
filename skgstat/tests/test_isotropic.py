import os
import pandas as pd
import skgstat as skg
from numpy.testing import assert_array_almost_equal


def _get_pan_sample():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'pan_sample.csv'))
    return df[['x', 'y']].values, df.z.values


def test_maxlag_change():
    # get data
    c, v = _get_pan_sample()

    # create a Variogram with default settings
    default = skg.Variogram(c, v)
    maxlag = skg.Variogram(c, v, maxlag=default.bins[-1])

    assert_array_almost_equal(
        default.experimental,
        maxlag.experimental,
        decimal=1
    )
