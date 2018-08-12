import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from skgstat import Variogram


class TestVariogram(unittest.TestCase):
    def setUp(self):
        # set up default values, whenever c and v are not important
        np.random.seed(42)
        self.c = np.random.gamma(10, 4, (30, 2))
        np.random.seed(42)
        self.v = np.random.normal(10, 4, 30)

    def test_standard_settings(self):
        V = Variogram(self.c, self.v)

        for x, y in zip(V.parameters, [439.405, 281.969, 0]):
            self.assertAlmostEqual(x, y, places=3)

    def test_pass_median_maxlag_on_instantiation(self):
        np.random.seed(1312)
        c = np.random.gamma(5, 1, (50, 2))

        np.random.seed(1312)
        v = np.random.weibull(5, 50)

        V = Variogram(c, v, maxlag='median')

        for x, y in zip(V.parameters, [1.914077, 0.002782, 0]):
            self.assertAlmostEqual(x, y, places=6)

    def test_residuals(self):
        V = Variogram(self.c, self.v)
        assert_array_almost_equal(
            V.residuals,
            np.array([0.96, -1.19, -0.28, 2.61, -0.9,
                      -0.43, -0.1, -2.32, -8.61, 10.61]),
            decimal=2
        )

    def test_rmse(self):
        V = Variogram(self.c, self.v)

        for model, rmse in zip(
                ['spherical', 'gaussian', 'matern', 'stable'],
                [4.4968, 4.4878, 4.4905, 4.4878]
        ):
            V.set_model(model)
            self.assertAlmostEqual(V.rmse, rmse, places=4)

    def test_mean_residual(self):
        V = Variogram(self.c, self.v)

        for model, mr in zip(
            ['spherical', 'cubic', 'matern', 'stable'],
            [2.8006, 2.711, 2.7433, 2.7315]
        ):
            V.set_model(model)
            self.assertAlmostEqual(V.mean_residual, mr, places=4)

    def test_nrmse(self):
        V = Variogram(self.c, self.v, n_lags=15)

        for model, nrmse in zip(
            ['spherical', 'gaussian', 'stable', 'exponential'],
            [0.4751, 0.4784, 0.4598, 0.4695]
        ):
            V.set_model(model)
            self.assertAlmostEqual(V.nrmse, nrmse, places=4)

    def test_nrmse_r(self):
        V = Variogram(self.c, self.v, estimator='cressie')

        self.assertAlmostEqual(V.nrmse_r, 0.40796, places=5)


if __name__ == '__main__':
    unittest.main()
