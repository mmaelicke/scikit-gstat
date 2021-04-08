"""
"""

import unittest

import numpy as np

from skgstat.estimators import matheron, cressie, dowd, genton
from skgstat.estimators import minmax, percentile, entropy


class TestEstimator(unittest.TestCase):
    def setUp(self):
        pass

    def test_matheron(self):
        # extract actual estimator
        e = matheron.py_func
        np.random.seed(42)

        self.assertAlmostEqual(
            e(np.random.normal(0, 1, 10000)),
            0.50342,
            places=6
        )

    def test_matheron_nan(self):
        # extract actual estimator
        e = matheron.py_func

        self.assertTrue(np.isnan(e(np.array([]))))

    def test_cressie(self):
        # extract actual estimator
        e = cressie.py_func

        np.random.seed(42)

        self.assertAlmostEqual(
            e(np.random.gamma(10, 4, 10000)),
            1686.7519,
            places=4
        )

    def test_cressie_nan(self):
        # extract actual estimator
        e = cressie.py_func

        self.assertTrue(np.isnan(e(np.array([]))))

    def test_dowd(self):
        np.random.seed(1306)
        x1 = np.random.weibull(14, 1000)
        np.random.seed(1312)
        x2 = np.random.gamma(10, 4, 100)

        # test
        self.assertAlmostEqual(dowd(x1), 2.0873, places=4)
        self.assertAlmostEqual(dowd(x2), 3170.97, places=2)

    def test_genton(self):
        # extract actual estimator
        e = genton.py_func

        np.random.seed(42)
        x1 = np.random.gamma(40, 2, 100)
        np.random.seed(42)
        x2 = np.random.gamma(30, 5, 1000)

        self.assertAlmostEqual(e(x1), 0.0089969, places=7)
        self.assertAlmostEqual(e(x2), 0.0364393, places=7)

    def test_genton_nan(self):
        # extract actual estimator
        e = genton.py_func

        # genton cannot be solved for only one element
        self.assertTrue(np.isnan(e(np.array([0.1]))))

    def test_minmax_skew(self):
        # heavily skewed gamma
        np.random.seed(1306)
        x = np.random.gamma(15, 20, 100)
        self.assertAlmostEqual(minmax(x), 1.5932, places=4)

    def test_minmax_pow(self):
        # L-stable pareto
        np.random.seed(2409)
        x = np.random.pareto(2, 10)
        self.assertAlmostEqual(minmax(x), 2.5, places=2)

    def test_percentile(self):
        np.random.seed(42)
        x = np.abs(np.random.normal(0, 1, 100000))

        self.assertAlmostEqual(percentile(x), 0.67588, places=5)
        self.assertAlmostEqual(percentile(x, 20), 0.25277, places=5)

    def test_entropy_default_bins(self):
        np.random.seed(42)
        x = np.random.normal(5, 1, 10000)

        self.assertAlmostEqual(entropy(x, bins=None), 3.0, places=2)

    def test_entropy_custom_bins(self):
        np.random.seed(123456789)
        x = np.random.gamma(10, 5, 10000)

        # custom bins
        self.assertAlmostEqual(
            entropy(x, [5, 15, 50, 51, 52, 53, 54, 55, 56, 100, 120, 150]),
            1.82, places=2
        )

        # default bins
        self.assertAlmostEqual(entropy(x), 2.91, places=2)


if __name__ == '__main__':
    unittest.main()
