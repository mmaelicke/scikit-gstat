"""
PyUnit Tests for the geostat.distance functions

TODO
Test different Parameter Combinations
"""

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
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


class TestDistance(unittest.TestCase):

    def test_Pointdist(self):
        """
        Testing PointDist Function
        """

        coordinates = [(0, 0), (0, 1), (1, 0), (1, 1), (3, 3)]
        assert_array_almost_equal(np.asarray(point_dist(coordinates)), result_2D)

    def test_euclidean_dist2D(self):
        """
        Test euclidean_dist2D function
        """

        coordinates = [(0, 0), (0, 1), (1, 0), (1, 1), (3, 3)]
        assert_array_almost_equal(np.asarray(nd_dist(coordinates)), result_2D)

    def test_euclidean_distND(self):
        """
        Test euclidean_distND function
        """

        coordinates = [(0, 0, 0, 0), (5, 5, 5, 5), (8, 8, 8, 8), (3, 3, 3, 3)]
        assert_array_almost_equal(np.asarray(nd_dist(coordinates)), result_ND)


if __name__ == '__main__':
    unittest.main()
