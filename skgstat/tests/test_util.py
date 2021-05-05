import pytest
import os
import numpy as np
import pandas as pd

from skgstat import Variogram
from skgstat.util import shannon_entropy
from skgstat.util.cross_validation import jacknife


# read the sample data
def get_sample() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'pan_sample.csv'))
    return df


def test_shannon_entropy():
    np.random.seed(42)

    # calculate the entropy the
    x = np.random.gamma(10, 15, size=1000)
    h = shannon_entropy(x, bins=15)

    assert np.abs(h - 2.943) < 0.001


def test_jacknife():
    # load the data sample
    df = get_sample()

    # create a Variogram
    V = Variogram(df[['x', 'y']].values, df.z.values, model='exponential', n_lags=25)

    rmse = V.cross_validate(n=30, seed=42)

    assert rmse - 16.623 < 0.1


def test_jackknife_metrics():
    # load the data sample
    df = get_sample()

    # create a Variogram
    V = Variogram(df[['x', 'y']].values, df.z.values, model='exponential', n_lags=25)

    rmse = jacknife(V, n=50, metric='RMSE', seed=1312)
    mse = jacknife(V, n=50, metric='MSE', seed=1312)

    assert np.sqrt(mse) - rmse < 0.001

    mae = jacknife(V, n=50, metric='MAE', seed=13062018)

    assert mae - 6.092 < 0.1


def test_unknown_cross_validation():
    # load the data sample
    df = get_sample()

    # create a Variogram
    V = Variogram(df[['x', 'y']].values, df.z.values, model='exponential', n_lags=25)

    with pytest.raises(AttributeError) as e:
        V.cross_validate(method='foobar')

    assert "'foobar' is not implemented" in str(e.value)
