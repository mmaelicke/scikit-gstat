"""

"""
import unittest

import numpy as np

from skgstat.models import spherical, exponential
from skgstat.models import gaussian, cubic, stable, matern


class TestModels(unittest.TestCase):
    def setUp(self):
        self.h = np.array([5, 10, 30, 50, 100])

    def test_spherical_default(self):
        result = [13.75, 20.0, 20.0, 20.0, 20.0]
        model = list(map(spherical, self.h, [10]*5, [20]*5))

        for r, m in zip(result, model):
            self.assertAlmostEqual(r, m, places=2)

    def test_spherical_nugget(self):
        result = [15.44, 27.56, 33.0, 34.0, 35.0]

        # calculate
        nuggets = [1, 2, 3, 4, 5]
        model = list(map(spherical, self.h, [15] * 5, [30] * 5, nuggets))

        for r, m in zip(result, model):
            self.assertAlmostEqual(r, m, places=2)

    def test_exponential_default(self):
        result = [5.18, 9.02, 16.69, 19., 19.95]
        model = list(map(exponential, self.h, [50]*5, [20]*5))

        for r, m in zip(result, model):
            self.assertAlmostEqual(r, m, places=2)

    def test_exponential_nugget(self):
        result = [7.64, 13.8, 26.31, 31.54, 34.8]

        # calculate
        nuggets = [1, 2, 3, 4, 5]
        model = list(map(exponential, self.h, [60] * 5, [30] * 5, nuggets))

        for r, m in zip(result, model):
            self.assertAlmostEqual(r, m, places=2)
