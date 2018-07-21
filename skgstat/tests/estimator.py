"""
"""

import unittest

import numpy as np

from skgstat.estimators import matheron


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


if __name__ == '__main__':
    unittest.main()