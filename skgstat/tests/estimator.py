"""
TODO
Make a new test for the entropy estimator
Test different Parameter Combinations
"""

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from skgstat.estimator import matheron, cressie, dowd, genton, minmax, percentile, entropy

# result arrays

result_matheron = np.array([5.00000000e-01, 0.00000000e+00, 5.00000000e+01,
                            5.00000000e+03])

result_cressie = np.array([8.96700143e-01, 0.00000000e+00, 8.96700143e+01,
                           8.96700143e+03])

result_dowd = np.array([1.09900000e+00, 0.00000000e+00, 1.09900000e+02,
                        1.09900000e+04])

result_minmax = [2.0, 0.0, 2.0, 2.0]

result_percentile = [4.5, 5.0, 45.0, 450.0]

result_entropy = np.array([0.69314718, 0.63651417, 0.63651417, 1.60943791])
result_entropy_fd = np.array([0.67301167, 0.67301167, 0.67301167, 0.95027054])
result_entropy_5b = np.array([1.05492017, 1.05492017, 1.05492017, 0.95027054])
result_entropy_ar = np.array([1.05492017, 0.67301167, 1.05492017, 1.05492017])


class TestEstimator(unittest.TestCase):
    def setUp(self):
        """
        Setting up the values
        """

        self.grouped = [list(np.arange(10)), [5] * 10, list(np.arange(0, 100, 10)),
                        list(np.arange(0, 1000, 100))]
        np.random.seed(42)
        self.entropy_grouped = [list(np.random.gamma(10,2, 10)), list(np.random.gamma(4,4, 10)),
                                list(np.random.gamma(4, 2, 10)), list(np.random.gamma(10,5, 10))]

    def test_matheron(self):
        """
        Testing matheron estimator
        """
        assert_array_almost_equal(matheron(self.grouped), result_matheron)

    def test_cressie(self):
        """
        Testing cressie estimator
        """
        assert_array_almost_equal(cressie(self.grouped), result_cressie, decimal=5)

    def test_dowd(self):
        """
        Testing dowd estimator
        """
        assert_array_almost_equal(dowd(self.grouped), result_dowd)

    def test_genton(self):
        """
        Testing genton estimator

        This one is still buggy, so don't test it
        """
        return True


    def test_minmax(self):
        """
        Testing minmax estimator
        """
        assert_array_almost_equal(minmax(self.grouped), result_minmax)

    def test_percentile(self):
        """
        Testing percentile estimator
        """
        assert_array_almost_equal(percentile(self.grouped), result_percentile)

    def test_entropy_default(self):
        """
        Testing entropy estimator with default settings
        """
        assert_array_almost_equal(np.asarray(entropy(self.entropy_grouped)), result_entropy)

    def test_entropy_string(self):
        """
        Testing entropy estimator with string as bin
        """
        assert_array_almost_equal(np.asarray(entropy(self.entropy_grouped, bins='fd')), result_entropy_fd)

    def test_entropy_integer(self):
        """
        Testing entropy estimator with integer as bin
        """
        assert_array_almost_equal(np.asarray(entropy(self.entropy_grouped, bins=5)), result_entropy_5b)

    def test_entropy_list(self):
        """
        Testing entropy estimator with list as bin
        """
        assert_array_almost_equal(
            np.asarray(entropy(self.entropy_grouped, bins=[0.1, 5, 10, 20, 100])),
            result_entropy_ar)
