"""Distance parameters must be shared by kriging observations and queries."""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from skgstat import MetricSpace, OrdinaryKriging, Variogram


class TestKrigingMetricParameters(unittest.TestCase):
    def test_parameterized_distances_match_scaled_coordinates(self):
        coordinates = np.array([[0., 0.], [1, 0], [0, 5], [1, 10], [4, 2], [5, 8]])
        values = np.array([0., 1., 4., 5., 8., 9.])
        targets = np.array([[.5, 4.], [2, 8.]])
        variance = np.array([1., 100.])
        euclidean = Variogram(coordinates / np.sqrt(variance), values,
                             fit_method='manual', fit_range=10., fit_sill=1.)
        settings = dict(min_points=3, max_points=3, solver='numpy')
        reference = OrdinaryKriging(euclidean, **settings)
        expected = reference.transform(*(targets / np.sqrt(variance)).T)
        metrics = [('seuclidean', {'V': variance}),
                   ('minkowski', {'p': 2., 'w': 1. / variance}),
                   ('mahalanobis', {'VI': np.diag(1. / variance)})]
        for metric, params in metrics:
            space = MetricSpace(coordinates, metric, dist_metric_kwargs=params)
            variogram = Variogram(space, values, dist_func=metric,
                                  fit_method='manual', fit_range=10., fit_sill=1.)
            for constructor in [{}, {'coordinates': space}, {'coordinates': coordinates}]:
                for explicit in [False, True]:
                    kriging = OrdinaryKriging(variogram, **constructor, **settings)
                    if explicit:
                        result = kriging.transform(MetricSpace(
                            targets, metric, dist_metric_kwargs=params))
                    else:
                        result = kriging.transform(*targets.T)
                    assert_allclose(result, expected)
                    assert_allclose(kriging.sigma, reference.sigma)

    def test_single_mahalanobis_target_with_supplied_inverse(self):
        target = np.array([[1., 2.]])
        space = MetricSpace(target, 'mahalanobis',
                            dist_metric_kwargs={'VI': np.eye(2)})
        self.assertEqual(space.dists.shape, (1, 1))
        assert_allclose(space.dists, 0.)
