"""

"""
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from skgstat.models import spherical, exponential
from skgstat.models import gaussian, cubic, stable, matern
from skgstat.models import variogram


class TestModels(unittest.TestCase):
    def setUp(self):
        self.h = np.array([5, 10, 30, 50, 100])

    def test_spherical_default(self):
        # extract the actual function
        f = spherical.py_func

        result = [13.75, 20.0, 20.0, 20.0, 20.0]

        model = list(map(f, self.h, [10]*5, [20]*5))

        for r, m in zip(result, model):
            self.assertAlmostEqual(r, m, places=2)

    def test_spherical_nugget(self):
        # extract the actual function
        f = spherical.py_func

        result = [15.44, 27.56, 33.0, 34.0, 35.0]

        # calculate
        nuggets = [1, 2, 3, 4, 5]
        model = list(map(f, self.h, [15] * 5, [30] * 5, nuggets))

        for r, m in zip(result, model):
            self.assertAlmostEqual(r, m, places=2)

    def test_exponential_default(self):
        # extract the actual function
        f = exponential.py_func

        result = [5.18, 9.02, 16.69, 19., 19.95]
        model = list(map(f, self.h, [50]*5, [20]*5))

        for r, m in zip(result, model):
            self.assertAlmostEqual(r, m, places=2)

    def test_exponential_nugget(self):
        # extract the actual function
        f = exponential.py_func

        result = [7.64, 13.8, 26.31, 31.54, 34.8]

        # calculate
        nuggets = [1, 2, 3, 4, 5]
        model = list(map(f, self.h, [60] * 5, [30] * 5, nuggets))

        for r, m in zip(result, model):
            self.assertAlmostEqual(r, m, places=2)

    def test_gaussian_default(self):
        # extract the actual function
        f = gaussian.py_func

        result = [0.96,  3.58, 16.62, 19.86, 20.]
        model = list(map(f, self.h, [45]*5, [20]*5))

        for r, m in zip(result, model):
            self.assertAlmostEqual(r, m, places=2)

    def test_gaussian_nugget(self):
        # extract the actual function
        f = gaussian.py_func

        result = [1.82,  5.15, 21.96, 32.13, 35.]

        # calculate
        nuggets = [1, 2, 3, 4, 5]
        model = list(map(f, self.h, [60] * 5, [30] * 5, nuggets))

        for r, m in zip(result, model):
            self.assertAlmostEqual(r, m, places=2)

    def _test_cubic_default(self):
        # extract the actual function
        f = cubic.py_func

        result = [6.13,  21.11,  88.12, 100., 100.]
        model = list(map(f, self.h, [50]*5, [100]*5))

        for r, m in zip(result, model):
            self.assertAlmostEqual(r, m, places=2)

    def test_cubic_nugget(self):
        # extract the actual function
        f = cubic.py_func

        result = [11.81, 34.74, 73., 74., 75.]

        # calculate
        nuggets = [1, 2, 3, 4, 5]
        model = list(map(f, self.h, [30] * 5, [70] * 5, nuggets))

        for r, m in zip(result, model):
            self.assertAlmostEqual(r, m, places=2)

    def test_stable_default(self):
        # extract the actual function
        f = stable.py_func

        result = [9.05, 23.53, 75.2, 95.02, 99.98]
        model = list(map(f, self.h, [50]*5, [100]*5, [1.5]*5))

        for r, m in zip(result, model):
            self.assertAlmostEqual(r, m, places=2)

    def test_stable_nugget(self):
        # extract the actual function
        f = stable.py_func

        result = [8.77, 10.8, 12.75, 13.91, 14.99]

        # calculate
        nuggets = [1, 2, 3, 4, 5]
        model = list(map(f, self.h, [20] * 5, [10] * 5, [0.5] * 5, nuggets))

        for r, m in zip(result, model):
            self.assertAlmostEqual(r, m, places=2)

    def test_matern_default(self):
        # extract the actual function
        f = matern.py_func

        result = [24.64, 43.2, 81.68, 94.09, 99.65]
        model = list(map(f, self.h, [50]*5, [100]*5, [0.50001]*5))

        for r, m in zip(result, model):
            self.assertAlmostEqual(r, m, places=2)

    def test_matern_nugget(self):
        # extract the actual function
        f = matern.py_func

        result = [3.44, 8.52, 12.99, 14., 15.]

        # calculate
        nuggets = [1, 2, 3, 4, 5]
        model = list(map(f, self.h, [20] * 5, [9.99999] * 5, [8] * 5, nuggets))

        for r, m in zip(result, model):
            self.assertAlmostEqual(r, m, places=2)
    
    def test_matern_r_switch(self):
        # run the default with an extreme s value

        # extract the actual function
        f = matern.py_func

        result = [24.64, 43.20, 81.68, 94.09, 99.65]

        model = list(map(f, self.h, [50]*5, [100]*5, [0.5]*5))

        assert_array_almost_equal(result, model, decimal=2)


class TestVariogramDecorator(unittest.TestCase):
    def test_scalar(self):
        @variogram
        def scalar_function(a, b):
            return a, b

        a, b = 1, 4
        self.assertEqual(scalar_function(1, 4), (a, b))

    def test_list(self):
        @variogram
        def adder(l, a):
            return l + a

        res = [5, 8, 12]

        for r, c in zip(res, adder([1, 4, 8], 4)):
            self.assertEqual(r, c)


if __name__=='__main__':
    unittest.main()
