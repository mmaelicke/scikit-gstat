import unittest

import numpy as np

from skgstat import Variogram


class TestVariogram(unittest.TestCase):
    def test_standard_settings(self):
        # check the parameters
        np.random.seed(42)
        c = np.random.gamma(10, 4, (30, 2))
        np.random.seed(42)
        v = np.random.normal(10, 4, 30)

        V = Variogram(c, v)

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


if __name__ == '__main__':
    unittest.main()
