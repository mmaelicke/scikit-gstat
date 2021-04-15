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


def test_sparse_matrix_no_warning():
    # make a really sparse matrix
    sparse = skg.MetricSpace(rcoords, max_dist=5)

    # call triangular_distance_matrix without warning
    V = skg.Variogram(sparse, rvals)
    V.triangular_distance_matrix
    

def test_dense_matrix_warning():
    dense = skg.MetricSpace(rcoords)

    # check the warning
    with pytest.raises(RuntimeWarning) as w:
        V = skg.Variogram(dense, rvals)
        V.triangular_distance_matrix

        assert 'Only available' in w.value
