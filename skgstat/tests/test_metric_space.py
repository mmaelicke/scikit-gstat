import pytest
import numpy as np
import skgstat as skg

# produce a random dataset
np.random.seed(42)
rcoords = np.random.gamma(40, 10, size=(500, 2))
np.random.seed(42)
rvals = np.random.normal(10, 4, 500)


def test_invalid_dist_func():
    # instantiate metrix space
    ms = skg.MetricSpace(rcoords, dist_metric='euclidean')

    with pytest.raises(AttributeError) as e:
        skg.Variogram(ms, rvals, dist_func='cityblock')

        assert 'Distance metric' in e.value
