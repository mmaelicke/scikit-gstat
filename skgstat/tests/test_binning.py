import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from skgstat.binning import even_width_lags, uniform_count_lags


class TestEvenWidth(unittest.TestCase):
    @staticmethod
    def test_normal():
        np.random.seed(42)
        assert_array_almost_equal(
            even_width_lags(np.random.normal(5, 1, 1000), 4, None),
            np.array([2.21318287, 4.42636575, 6.63954862, 8.85273149])
        )

    @staticmethod
    def test_more_bins():
        np.random.seed(42)
        assert_array_almost_equal(
            even_width_lags(np.random.normal(5, 1, 1000), 10, None),
            np.array([0.88527315, 1.7705463, 2.65581945, 3.5410926, 4.42636575,
                      5.3116388, 6.19691204, 7.08218519, 7.96745834, 8.8527314])
        )

    @staticmethod
    def test_maxlag():
        np.random.seed(42)
        assert_array_almost_equal(
            even_width_lags(np.random.normal(5, 1, 1000), 4, 4.4),
            np.array([1.1, 2.2, 3.3, 4.4])
        )

    @staticmethod
    def test_too_large_maxlag():
        np.random.seed(42)
        assert_array_almost_equal(
            even_width_lags(np.random.normal(5, 1, 1000), 4, 400),
            np.array([2.21318287, 4.42636575, 6.63954862, 8.85273149])
        )

    @staticmethod
    def test_median_split():
        np.random.seed(42)
        assert_array_almost_equal(
            even_width_lags(np.random.normal(5, 1, 1000), 2, None),
            np.array([4.42636575, 8.85273149])
        )


class TestUniformCount(unittest.TestCase):
    def test_normal(self):
        np.random.seed(42)
        assert_array_almost_equal(
            uniform_count_lags(np.random.normal(10, 2, 1000), 4, None),
            np.array([8.7048, 10.0506, 11.2959, 17.7055]),
            decimal=4
        )

# TODO: put the other test for binning here


if __name__ == '__main__':
    unittest.main()
