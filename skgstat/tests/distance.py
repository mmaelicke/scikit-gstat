"""
PyUnit Tests for the geostat.distance functions

TODO
Test different Parameter Combinations
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose
from skgstat.distance import point_dist, nd_dist

# Get the result arrays
result_2D = np.array([[0., 1., 1., 1.41421356, 4.24264069],
                      [1., 0., 1.41421356, 1., 3.60555128],
                      [1., 1.41421356, 0., 1., 3.60555128],
                      [1.41421356, 1., 1., 0., 2.82842712],
                      [4.24264069, 3.60555128, 3.60555128, 2.82842712, 0.]])

result_ND = np.array([[0., 10., 16., 6.],
                      [10., 0., 6., 4.],
                      [16., 6., 0., 10.],
                      [6., 4., 10., 0.]])

rank_2D = np.matrix([[3., 9.5, 9.5, 15.5, 24.5, 9.5, 3., 15.5, 9.5,
                      21.5, 9.5, 15.5, 3., 9.5, 21.5, 15.5, 9.5, 9.5,
                      3., 18.5, 24.5, 21.5, 21.5, 18.5, 3.]])

rank_ND = np.matrix([[2.5, 12.5, 15.5, 8.5, 12.5, 2.5, 8.5, 5.5, 15.5,
                      8.5, 2.5, 12.5, 8.5, 5.5, 12.5, 2.5]])

class TestPointDist(unittest.TestCase):
    def setUp(self):
        self.coords = [(0, 0), (0, 1), (1, 0), (1, 1), (3, 3)]

    def test_point_dist(self):
        """
        Test point_dist function with 'euclidean' metric.
        """
        assert_allclose(np.asarray(point_dist(self.coords)), result_2D, atol=1e-5)

    def test_point_rank(self):
        """
        Test point_dist function with 'rank' metric.
        """
        assert_allclose(np.asarray(point_dist(self.coords, metric='rank')),
                        rank_2D, atol=0.1)

    def test_coordinate_check(self):
        """
        Use a not allowed coordinate.
        """
        coordinates = self.coords
        coordinates[2] = (1, 2, 3, 4)
        with self.assertRaises(ValueError):
            point_dist(coordinates, metric='euclidean')

    def test_unknown_metric(self):
        """
        Use a not known metric
        """
        with self.assertRaises(ValueError):
            point_dist(self.coords, metric='I am not a metric')


class TestNdDist(unittest.TestCase):
    def setUp(self):
        self.coords2d = [(0, 0), (0, 1), (1, 0), (1, 1), (3, 3)]
        self.coords = [(0, 0, 0, 0), (5, 5, 5, 5), (8, 8, 8, 8), (3, 3, 3, 3)]

    def test_euclidean_dist2D(self):
        """
        Test nd_dist function on 2D data with 'euclidean' metric.
        """
        assert_allclose(np.asarray(nd_dist(self.coords2d)), result_2D, atol=0.1)

    def test_euclidean_distND(self):
        """
        Test nd_dist function on n-dimensional data with 'euclidean' metric.
        """
        assert_allclose(np.asarray(nd_dist(self.coords)), result_ND, atol=0.1)

    def test_rank_dist2D(self):
        """
        Test nd_dist function on 2D data with 'rank' metric.
        """
        assert_allclose(np.asarray(nd_dist(self.coords2d, metric='rank')), rank_2D,
                        atol=0.1)

    def test_rank_distND(self):
        """
        Test nd_dist function on n-dimensional data with 'rank' metric.
        """
        assert_allclose(np.asarray(nd_dist(self.coords, metric='rank')),
                        rank_ND, atol=0.1)

    def test_misshaped_coordinates(self):
        """
        Mishape a coordinate, then a ValueError should get fired
        """
        coordinates = self.coords
        coordinates[1] = (1,2)    # not enough dimensions
        with self.assertRaises(ValueError):
            nd_dist(coordinates)

    def test_unknown_metric(self):
        """
        Use a unknown metric
        """
        with self.assertRaises(ValueError):
            nd_dist(self.coords, metric='I am not a metric')



if __name__ == '__main__':
    unittest.main()
