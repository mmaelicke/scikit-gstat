"""
TODO
Make a new test for the entropy estimator
Test different Parameter Combinations
"""

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from skgstat.estimator import matheron, cressie, dowd, genton, minmax, entropy

# result arrays

result_matheron = np.array([5.00000000e-01, 0.00000000e+00, 5.00000000e+01,
                            5.00000000e+03])

result_cressie = np.array([8.96700143e-01, 0.00000000e+00, 8.96700143e+01,
                           8.96700143e+03])

result_dowd = np.array([1.09900000e+00, 0.00000000e+00, 1.09900000e+02,
                        1.09900000e+04])

result_genton = np.array([2.46198050e+02, 2.46198050e+02, 2.46198050e+04,
                          2.46198050e+06])

result_minmax = [2.0, 0.0, 2.0, 2.0]

result_entropy = np.array([0., 0., 0., 0.])


class TestEstimator(unittest.TestCase):
    def setUp(self):
        """
        Setting up the values
        """

        self.grouped = [list(np.arange(10)), [5] * 10, list(np.arange(0, 100, 10)),
                                  list(np.arange(0, 1000, 100))]

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
        """

        assert_array_almost_equal(genton(self.grouped), result_genton)


    def test_minmax(self):
        """
        Testing minmax estimator
        """

        assert_array_almost_equal(minmax(self.grouped), result_minmax)


    def test_entropy(self):
        """
        Testing entropy estimator
        """

        assert_array_almost_equal(np.asarray(entropy(self.grouped)), result_entropy)
