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


def test_unkonwn_metric():
    with pytest.raises(ValueError) as e:
        skg.MetricSpace(rcoords, dist_metric='foobar')

        assert 'Unknown Distance Metric:' in e.value


def test_tree_non_euklidean():
    with pytest.raises(ValueError) as e:
        ms = skg.MetricSpace(rcoords, 'cityblock')
        ms.tree

        assert 'can only be constructed' in e.value


def test_metric_pair_metrix():
    c1 = np.random.gamma(100, 4, (300, 2))
    c2 = np.random.gamma(50, 5, (100, 2))
    ms1 = skg.MetricSpace(c1, dist_metric='cityblock')
    ms2 = skg.MetricSpace(c2, dist_metric='euclidean')

    with pytest.raises(ValueError) as e:
        skg.MetricSpacePair(ms1, ms2)

        assert 'same distance metric' in e.value


def test_metric_pair_max_dist():
    c1 = np.random.gamma(100, 4, (300, 2))
    c2 = np.random.gamma(50, 5, (100, 2))
    ms1 = skg.MetricSpace(c1, max_dist=50)
    ms2 = skg.MetricSpace(c2, max_dist=400)

    with pytest.raises(ValueError) as e:
        skg.MetricSpacePair(ms1, ms2)

        assert 'same max_dist' in e.value
