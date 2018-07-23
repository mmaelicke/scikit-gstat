"""
"""

import unittest

import numpy as np

from skgstat.estimators import matheron, cressie


class TestEstimator(unittest.TestCase):
    def setUp(self):
        pass

    def test_matheron(self):
        np.random.seed(42)

        self.assertAlmostEqual(
            matheron(np.random.normal(0, 1, 10000)),
            0.50342,
            places=6
        )

    def test_matheron_nan(self):
        self.assertTrue(np.isnan(matheron(np.array([]))))

    def test_cressie(self):
        np.random.seed(42)

        self.assertAlmostEqual(
            cressie(np.random.gamma(10, 4, 10000)),
            1686.7519,
            places=4
        )


if __name__ == '__main__':
    unittest.main()
