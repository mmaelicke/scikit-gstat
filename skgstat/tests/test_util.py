import numpy as np

from skgstat.util import shannon_entropy


def test_shannon_entropy():
    np.random.seed(42)

    # calculate the entropy the
    x = np.random.gamma(10, 15, size=1000)
    h = shannon_entropy(x, bins=15)

    assert np.abs(h - 2.943) < 0.001
