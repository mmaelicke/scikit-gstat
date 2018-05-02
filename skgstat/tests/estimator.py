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

result_minmax = np.array([1.9927489681047741, 1.8267020635804991, 1.5392934574169328, 2.0360906123190832])

result_percentile_r = np.array([6.3486663708926443, 6.5981915461999279, 4.7812030455283168, 7.2582516292816663])
result_percentile = np.array([1.0, 0.0, 10.0, 100.0])

# entropies
result_uni_entropy = np.array([0.0081243, 0.0081243, 0.0081243, 0.0081243])
result_uni_entropy_fd = np.array([-1.44270225e-05, -1.44270225e-05, -1.44270225e-05,
                                  -1.44270225e-05])
result_uni_entropy_5b = np.array([0.00064996, 0.00064996, 0.00064996, 0.00064996])
result_uni_entropy_ar = np.array([0.00064996, 0.00064996, 0.00064996, 0.00064996])

result_entropy = np.array([1.9295937, 2.32944639, 2.32944639, 2.32944639])
result_entropy_fd = np.array([1.92211936, 1.37096112, 0.97094233, 1.37096112])
result_entropy_5b = np.array([1.92211936, 1.92211936, 1.92211936, 1.92211936])
result_entropy_ar = np.array([1.92211936, 1.52226666, 0.97144062, 1.52226666])

class TestEstimator(unittest.TestCase):
    def setUp(self):
        """
        Setting up the values
        """

        self.grouped = [list(np.arange(10)), [5] * 10, list(np.arange(0, 100, 10)),
                        list(np.arange(0, 1000, 100))]
        np.random.seed(42)
        self.random_grouped = [list(np.random.gamma(10,2, 10)), list(np.random.gamma(4,4, 10)),
                                list(np.random.gamma(4, 2, 10)), list(np.random.gamma(10,5, 10))]

    def test_matheron(self):
        """
        Testing matheron estimator
        """
        assert_array_almost_equal(matheron(self.grouped), result_matheron)

    def test_matheron_uneven_input(self):
        """
        Raise the ValueError on uneven length input
        """
        with self.assertRaises(ValueError):
            matheron([0, 1, 2])

    def test_matheron_nan_on_zerodivision(self):
        """
        return nan on empty input
        """
        self.assertTrue(np.isnan(matheron([])))

    def test_cressie(self):
        """
        Testing cressie estimator
        """
        assert_array_almost_equal(cressie(self.grouped), result_cressie,
                                  decimal=5)

    def test_cressie_uneven_input(self):
        """
        Raise the ValueError on uneven length input
        """
        with self.assertRaises(ValueError):
            cressie([0, 1, 2])

    def test_cressie_nan_on_zerodivision(self):
        """
        return nan on empty input
        """
        self.assertTrue(np.isnan(cressie([])))

    def test_dowd(self):
        """
        Testing dowd estimator
        """
        assert_array_almost_equal(dowd(self.grouped), result_dowd)

    def test_dowd_uneven_input(self):
        """
        Raise the ValueError on uneven length input
        """
        with self.assertRaises(ValueError):
            dowd([0, 1, 2])

    def test_genton_single_value(self):
        """
        Testing genton estimator with a single value calculation

        """
        np.random.seed(42)
        self.assertAlmostEqual(
            genton(np.random.normal(4, 2, size=20)),
            1.97039693,
            places=8
        )

    def test_genton_large_dataset(self):
        """
        Testing genton on large dataset
        """
        np.random.seed(42)
        self.assertAlmostEqual(
            genton(np.random.normal(4,2, size=1000)),
            0.0329977,
            places=8
        )


    def test_minmax(self):
        """
        Testing minmax estimator
        """
        assert_array_almost_equal(minmax(self.random_grouped), result_minmax)

    def test_minmax_uneven_input(self):
        """
        Raise the ValueError on uneven length input
        """
        with self.assertRaises(ValueError):
            minmax([0, 1, 2])

    def test_percentile_random(self):
        """
        Testing percentile estimator on randomized data
        """
        assert_array_almost_equal(percentile(self.random_grouped),
                                  result_percentile_r)

    def test_percentile_grouped(self):
        """
        Testing percentile estimator on grouped data
        """
        assert_array_almost_equal(percentile(self.grouped), result_percentile)

    def test_percentile_uneven_input(self):
        """
        Raise the ValueError on uneven length input
        """
        with self.assertRaises(ValueError):
            percentile([0, 1, 2])

    def test_entropy_uniform_default(self):
        """
        Testing the entropy estimator on uniform distributions,
        with and without gaps

        """
        assert_array_almost_equal(entropy(self.grouped), result_uni_entropy)

    def test_entropy_uniform_string(self):
        """
        Testing entropy estimator with string as bin on uniform distributions
        """
        assert_array_almost_equal(
            np.asarray(entropy(self.grouped, bins='fd')),
            result_uni_entropy_fd
        )

    def test_entropy_uniform_integer(self):
        """
        Testing entropy estimator with integer as bin on uniform distributions
        """
        assert_array_almost_equal(
            np.asarray(entropy(self.grouped, bins=5)),
            result_uni_entropy_5b
        )

    def test_entropy_uniform_list(self):
        """
        Testing entropy estimator with list as bin on uniform distributions
        """
        assert_array_almost_equal(
            np.asarray(entropy(self.grouped, bins=[0, 0.1, 5, 10, 20, 100])),
            result_uni_entropy_ar
        )

    def test_entropy_default(self):
        """
        Testing entropy estimator with default settings
        """
        assert_array_almost_equal(
            np.asarray(entropy(self.random_grouped)),
            result_entropy
        )

    def test_entropy_string(self):
        """
        Testing entropy estimator with string as bin
        """
        assert_array_almost_equal(
            np.asarray(entropy(self.random_grouped, bins='fd')),
            result_entropy_fd
        )

    def test_entropy_integer(self):
        """
        Testing entropy estimator with integer as bin
        """
        assert_array_almost_equal(
            np.asarray(entropy(self.random_grouped, bins=5)),
            result_entropy_5b
        )

    def test_entropy_list(self):
        """
        Testing entropy estimator with list as bin
        """
        assert_array_almost_equal(
            np.asarray(
                entropy(self.random_grouped, bins=[0, 0.1, 5, 10, 20, 100])
            ),
            result_entropy_ar
        )

    def test_entropy_uneven_input(self):
        """
        Raise the ValueError on uneven length input
        """
        with self.assertRaises(ValueError):
            entropy([0, 1, 2])

    def test_entropy_single_value(self):
        """
        Calculate a single value with bins=None.
        """
        np.random.seed(42)
        self.assertAlmostEqual(
            entropy(np.random.normal(4, 2, size=150)),
            3.412321, places=6
        )

        np.random.seed(1337)
        self.assertNotAlmostEqual(
            entropy(np.random.normal(4, 2, size=150)),
            3.412321, places=6
        )


if __name__ == '__main__':
    unittest.main()