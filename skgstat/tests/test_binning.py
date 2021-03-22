import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from skgstat.binning import (
    even_width_lags,
    uniform_count_lags,
    auto_derived_lags,
    kmeans,
    ward,
    stable_entropy_lags
)


class TestEvenWidth(unittest.TestCase):
    @staticmethod
    def test_normal():
        np.random.seed(42)
        bins, _ = even_width_lags(np.random.normal(5, 1, 1000), 4, None)

        assert_array_almost_equal(
            bins,
            np.array([2.21318287, 4.42636575, 6.63954862, 8.85273149])
        )

    @staticmethod
    def test_more_bins():
        np.random.seed(42)

        bins, _ = even_width_lags(np.random.normal(5, 1, 1000), 10, None)

        assert_array_almost_equal(
            bins,
            np.array([0.88527315, 1.7705463, 2.65581945, 3.5410926, 4.42636575,
                      5.3116388, 6.19691204, 7.08218519, 7.96745834, 8.8527314])
        )

    @staticmethod
    def test_maxlag():
        np.random.seed(42)
        bins, _ = even_width_lags(np.random.normal(5, 1, 1000), 4, 4.4)

        assert_array_almost_equal(
            bins,
            np.array([1.1, 2.2, 3.3, 4.4])
        )

    @staticmethod
    def test_too_large_maxlag():
        np.random.seed(42)

        bins, n = even_width_lags(np.random.normal(5, 1, 1000), 4, 400)
        assert_array_almost_equal(
            bins,
            np.array([2.21318287, 4.42636575, 6.63954862, 8.85273149])
        )

    @staticmethod
    def test_median_split():
        np.random.seed(42)
        bins, _ = even_width_lags(np.random.normal(5, 1, 1000), 2, None)
        assert_array_almost_equal(
            bins,
            np.array([4.42636575, 8.85273149])
        )


class TestUniformCount(unittest.TestCase):
    def test_normal(self):
        np.random.seed(42)

        bins, _ = uniform_count_lags(np.random.normal(10, 2, 1000), 4, None)
        assert_array_almost_equal(
            bins,
            np.array([8.7048, 10.0506, 11.2959, 17.7055]),
            decimal=4
        )


class TestDerivedBins(unittest.TestCase):
    def test_auto(self):
        np.random.seed(42)

        bins, n = auto_derived_lags(
            np.random.normal(10, 2, 1000),
            'sturges',
            None
        )

        # sturges should find 11 classes
        self.assertTrue(n == 11)

        assert_array_almost_equal(
            bins,
            np.array([4.8, 6.1, 7.4, 8.7, 10., 11.3, 12.5, 13.8, 15.1, 16.4, 17.7]),
            decimal=1
        )

    def test_skewed(self):
        np.random.seed(1312)

        bins, n = auto_derived_lags(np.random.gamma(10, 20, 500), 'doane', 100)

        # doane should condense to 6 here
        self.assertTrue(n == 6)

        assert_array_almost_equal(
            bins,
            np.array([75.6, 80.4, 85.3, 90.2, 95.1, 100.]),
            decimal=1
        )


class TestClusteringBins(unittest.TestCase):
    def test_kmeans(self):
        np.random.seed(1312)
        bins, _ = kmeans(np.random.gamma(10, 40, 500), 6, None)

        assert_array_almost_equal(
            np.array([118.5, 283.2, 374.7, 467.9, 574.5, 762.5]),
            bins,
            decimal=1
        )

    def test_ward(self):
        np.random.seed(1312)
        bins, _ = ward(np.random.gamma(10, 40, 500), 6, None)

        assert_array_almost_equal(
            np.array([122.7, 283.7, 352.3, 422.7, 520.9, 660.4]),
            bins,
            decimal=1
        )

    def test_ward_median(self):
        np.random.seed(1312)
        bins, _ = ward(np.random.gamma(10, 40, 500), 6, None, binning_agg_func='median')

        assert_array_almost_equal(
            np.array([126.2, 287.8, 354.6, 421.7, 517., 643.1]),
            bins,
            decimal=1
        )

    def test_stable_entropy(self):
        np.random.seed(1312)
        d = np.random.gamma(1500, 40, 100)

        # run
        bins, _ = stable_entropy_lags(d, 6, None)

        assert_array_almost_equal(
            np.array([10964.6, 21929.3, 32893.9, 43858.6, 54823.2, 69077.2]),
            bins,
            decimal=1
        )


if __name__ == '__main__':
    unittest.main()
