import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from skgstat import models, stmodels


class TestSumModel(unittest.TestCase):
    def setUp(self):
        # spatial range = 10;  spatial sill = 5
        self.Vx = lambda h: models.spherical(h, 10, 5)
        # temporal range = 5;  temporal sill = 7
        self.Vt = lambda t: models.cubic(t, 5, 7)

        self.lags = np.array([
            [1.2, 1.0],
            [5.0, 2.5],
            [10., 5.0],
            [12., 7.0],
            [4.2, 7.0],
            [12.0, 3.4]
        ])

    def test_default(self):
        assert_array_almost_equal(
            [stmodels.sum(lag, self.Vx, self.Vt) for lag in self.lags],
            [2.37, 8.76, 12., 12., 9.96, 11.61],
            decimal=2
        )

    def test_default_as_array(self):
        assert_array_almost_equal(
            stmodels.sum(self.lags, self.Vx, self.Vt),
            [2.37, 8.76, 12., 12., 9.96, 11.61],
            decimal=2
        )


class TestProductModel(unittest.TestCase):
    def setUp(self):
        # spatial range = 10;  spatial sill = 5
        self.Vx = lambda h: models.spherical(h, 10, 5)
        # temporal range = 5;  temporal sill = 7
        self.Vt = lambda t: models.cubic(t, 5, 7)

        self.lags = np.array([
            [1.2, 1.0],
            [5.0, 2.5],
            [10., 5.0],
            [12., 7.0],
            [4.2, 7.0],
            [12.0, 3.4]
        ])

    def test_default(self):
        assert_array_almost_equal(
            [stmodels.product(h, self.Vx, self.Vt, 5, 7) for h in self.lags],
            [12.34, 32.37, 35., 35., 35., 35.],
            decimal=2
        )

    def test_default_as_array(self):
        assert_array_almost_equal(
            stmodels.product(self.lags, self.Vx, self.Vt, 5, 7),
            [12.34, 32.37, 35., 35., 35., 35.],
            decimal=2
        )


class TestProductSumModel(unittest.TestCase):
    def setUp(self):
        # spatial range = 10;  spatial sill = 5
        self.Vx = lambda h: models.spherical(h, 10, 5)
        # temporal range = 5;  temporal sill = 7
        self.Vt = lambda t: models.cubic(t, 5, 7)

        self.lags = np.array([
            [1.2, 1.0],
            [5.0, 2.5],
            [10., 5.0],
            [12., 7.0],
            [4.2, 7.0],
            [12.0, 3.4]
        ])

    def test_default(self):
        assert_array_almost_equal(
            [stmodels.product_sum(h, self.Vx, self.Vt,
                k1=2.2, k2=2.3, k3=4.3, Cx=5, Ct=7) for h in self.lags],
            [35.55, 101.99, 118.6, 118.6, 113.92, 116.91],
            decimal=2
        )

    def test_default_as_array(self):
        assert_array_almost_equal(
            stmodels.product_sum(self.lags, self.Vx, self.Vt,
                k1=2.2, k2=2.3, k3=4.3, Cx=5, Ct=7),
            [35.55, 101.99, 118.6, 118.6, 113.92, 116.91],
            decimal=2
        )

    def test_with_zero_ks(self):
        assert_array_almost_equal(
            stmodels.product_sum(self.lags, self.Vx, self.Vt,
                k1=0, k2=0, k3=0, Cx=5, Ct=7),
            [0., 0., 0., 0., 0., 0.],
            decimal=2
        )

    def test_with_all_one(self):
        assert_array_almost_equal(
            stmodels.product_sum(self.lags, self.Vx, self.Vt,
                k1=1, k2=1, k3=1, Cx=5, Ct=7),
            [14.71, 41.13, 47.  ,47.  ,44.96, 46.61],
            decimal=2
        )

    def test_as_product_model(self):
        assert_array_almost_equal(
            stmodels.product_sum(self.lags, self.Vx, self.Vt,
                k1=1, k2=0, k3=0, Cx=5, Ct=7),
            stmodels.product(self.lags, self.Vx, self.Vt, 5, 7),
            decimal=2
        )


class TestMetricModel(unittest.TestCase):
    def setUp(self):
        self.lags = np.array([
            [0., 0.],
            [3., 0.],
            [0., 1.],
            [6., 2.],
            [12., 4.]
        ])

    def test_default(self):
        # space range = 10, time range = 2.5, sill = 5
        assert_array_almost_equal(
            stmodels.metric(self.lags, models.spherical, 10, 2.5, 5),
            [0., 2.18, 2.84, 5., 5.],
            decimal=2
        )

    def test_with_nugget(self):
        assert_array_almost_equal(
            stmodels.metric(self.lags, models.spherical, 10, 2.5, 5, 1),
            [1., 3.18, 3.84, 6., 6.],
            decimal=2
        )

    def test_as_space_model(self):
        assert_array_almost_equal(
            stmodels.metric(self.lags[:2], models.stable, 10, 2.5, 5, 1.5),
            models.stable(self.lags[:2, 0], 10, 5, 1.5),
            decimal=2
        )

    def test_single_lag(self):
        self.assertAlmostEqual(
            stmodels.metric((0., 2.5), models.exponential, 10, 2.5, 5),
            models.exponential(10., 10, 5)
        )


if __name__ == '__main__':
    unittest.main()
