import pytest
import numpy as np
import skgstat as skg
import scipy

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


def test_unknown_metric():
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

def test_raster_metric():

    # Generate a gridded dataset
    shape = (100, 100)
    np.random.seed(42)
    vals = np.random.normal(0, 1, size=shape)

    # Coordinates
    x = np.arange(0, shape[0])
    y = np.arange(0, shape[1])
    xx, yy = np.meshgrid(x, y)

    # Flatten everything because we don't care about the 2D at this point
    coords = np.dstack((xx.flatten(), yy.flatten())).squeeze()
    vals = vals.flatten()

    # Run the computation
    rems = skg.RasterEquidistantMetricSpace(coords, shape=shape, extent=(x[0],x[-1],y[0],y[-1]), samples=10, runs=10,
                                            rnd=42, verbose=True)

    # Minimal check of the output
    assert rems.max_dist == pytest.approx(140,rel=0.01)
    assert rems.res == pytest.approx(1, rel=0.0001)
    assert isinstance(rems.dists, scipy.sparse.csr.csr_matrix)
    assert rems.dists.shape == (10000, 10000)

    # Check the random state provides the same final center
    assert all(rems._centers[-1] == np.array([62, 52]))

    # Check the interface with a Variogram object works
    V = skg.Variogram(rems, vals)

    assert V.bin_count is not None
    # Check the variogram is always the same with the random state given
    assert V.experimental[0] == pytest.approx(0.89,0.01)

    # Check that the routines are robust to very few data points in the grid (e.g., from nodata values)
    coords_sub = coords[0::1000]
    vals_sub = vals[0::1000]
    rems_sub = skg.RasterEquidistantMetricSpace(coords_sub, shape=shape, extent=(x[0],x[-1],y[0],y[-1]), samples=100, runs=10,
                                            rnd=42)
    V = skg.Variogram(rems_sub, vals_sub)

    # Check with a single isolated point possibly being used as center
    coords_sub = np.concatenate(([coords[0]], coords[-10:]))
    vals_sub = np.concatenate(([vals[0]], vals[-10:]))
    rems_sub = skg.RasterEquidistantMetricSpace(coords_sub, shape=shape, extent=(x[0],x[-1],y[0],y[-1]), samples=100, runs=11,
                                            rnd=42)
    V = skg.Variogram(rems_sub, vals_sub)
