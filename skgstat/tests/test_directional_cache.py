import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from skgstat import DirectionalVariogram


class TestDirectionalCache(unittest.TestCase):
    def setUp(self):
        x, y = np.meshgrid(np.arange(10.), np.arange(8.))
        self.c = np.column_stack((x.ravel(), y.ravel()))
        self.v = np.sin(x.ravel()) + y.ravel() / 3
        self.defaults = dict(azimuth=0, tolerance=45, bandwidth=3.25,
                             n_lags=5, maxlag=8)

    def check_change(self, initial, changes, **kwargs):
        options = dict(self.defaults, **kwargs)
        options.update(initial)
        var = DirectionalVariogram(self.c, self.v, **options)
        # Populate every dependent cache before changing the direction.
        var.bins
        var.bin_count
        var.parameters
        angles, distances = var._angles, var._euclidean_dist
        for key, value in changes.items():
            if key == 'directional_model':
                var.set_directional_model(value)
            else:
                setattr(var, key, value)
        self.assertIs(var._angles, angles)
        self.assertIs(var._euclidean_dist, distances)
        options.update(changes)
        fresh = DirectionalVariogram(self.c, self.v, **options)

        assert_array_equal(var.bins, fresh.bins)
        assert_array_equal(var.bin_count, fresh.bin_count)
        assert_array_equal(var.lag_groups(), fresh.lag_groups())
        assert_array_equal(var._direction_mask(), fresh._direction_mask())
        assert_allclose(var.experimental, fresh.experimental)
        assert_allclose(var.parameters, fresh.parameters)
        assert_allclose(var.cov, fresh.cov)
        assert_allclose(var.transform(np.linspace(0, 8, 20)),
                        fresh.transform(np.linspace(0, 8, 20)))
        # describe()/transform() can explicitly force a refit, which rebuilds
        # geometry. The directional setter itself must leave it reusable.
        assert_array_equal(var._angles, angles)
        assert_array_equal(var._euclidean_dist, distances)

    def test_direction_parameters(self):
        for changes in [dict(azimuth=90), dict(tolerance=120), dict(bandwidth=6)]:
            self.check_change({}, changes)

    def test_directional_models(self):
        self.check_change({}, dict(directional_model='compass'))

        def upper_half(angles, distances):
            return angles >= 0

        def lower_half(angles, distances):
            return angles < 0

        self.check_change(dict(directional_model=upper_half),
                          dict(directional_model=lower_half))

    def test_data_derived_bins(self):
        for bin_func in ['even', 'uniform', 'fd']:
            self.check_change({}, dict(azimuth=90, bandwidth=6),
                              bin_func=bin_func, maxlag=None)

    def test_explicit_edges_are_preserved(self):
        self.check_change({}, dict(azimuth=90, tolerance=90),
                          bin_func=np.array([1., 2., 4., 6., 8.]))


if __name__ == '__main__':
    unittest.main()
