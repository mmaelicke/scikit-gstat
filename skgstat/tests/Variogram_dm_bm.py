"""
TODO
Test different Parameter Combinations
"""

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
# from skgstat.signal.sample.rf import gaussian2D
from skgstat import Variogram
from skgstat.tests.distance import result_2D, result_ND
from skgstat.tests.binning import binify_even_bin, group_to_bin
import skgstat.tests.binning

bm_array = np.array([[0., 3., 6., 4., 0.],
                     [3., 0., 9., 5., 2.],
                     [6., 9., 0., 5., 7.],
                     [4., 5., 5., 0., 4.],
                     [0., 2., 7., 4., 0.]])


class TestVariogramClassBmDm(unittest.TestCase):
    def setUp(self):
        """
        Set up sample data from a gaussian random field

        rf = gaussian2D(seed=42)
        np.random.seed(42)
        x = np.random.randint(0, 100, 50)
        y = np.random.randint(0, 100, 50)

        self.values = rf[x, y]
        self.coordinates = list(zip(x, y))
        """

        self.coordinates = np.array([(0, 0), (1, 1), (2, 2), (3, 3)] * 5)
        self.values = np.array([0, 1, 2, 3] * 5)

    def test_Variogram_dm_2D(self):
        """
        Testing distance Matrix 2D with the Variogram class
        """

        coordinates = [(0, 0), (0, 1), (1, 0), (1, 1), (3, 3)]
        values = np.arange(5)
        assert_array_almost_equal(np.asarray(Variogram(coordinates, values).dm), result_2D)

    def test_Variogram_dm_ND(self):
        """
        Testing distance Matrix ND with the Variogram Class
        """

        coordinates = [(0, 0, 0, 0), (5, 5, 5, 5), (8, 8, 8, 8), (3, 3, 3, 3)]
        values = np.arange(4)
        assert_array_almost_equal(np.asarray(Variogram(coordinates, values).dm), result_ND)

    def test_Variogram_bm_even_width_bm(self):
        """
        Testing the binning Matrix with the Variogram Class
        """

        assert_array_almost_equal(Variogram(self.coordinates, self.values).bm,
                                  skgstat.tests.binning.result_binify_even_width_bm)

    def test_Variogram_bm_even_width_bw(self):
        """
        Testing the binning widths with the Variogram Class
        """

        var = Variogram(self.coordinates, self.values)
        var.set_bm()

        assert_array_almost_equal(var.bin_widths,
                                  skgstat.tests.binning.result_binify_even_width_bw)

    def test_Variogram_bm_even_width_bm_n6(self):
        """
        Testing the binning Matrix and six bins with the Variogram Class
        """

        assert_array_almost_equal(Variogram(self.coordinates, self.values, N=6).bm,
                                  skgstat.tests.binning.result_binify_even_width_bm_n6)

    def test_Variogram_bm_even_width_bw_n6(self):
        """
        Testing the binning Matrix with 6 bins with the Variogram Class
        """

        var = Variogram(self.coordinates, self.values, N=6)

        assert_array_almost_equal(var.bin_widths,
                                  skgstat.tests.binning.result_binify_even_width_bw_n6)

    def test_Variogram_bm_with_dm(self):
        """
        Testing binning matrix of binify_even_width Function with a distance matrix in Variogram Class
        """

        var = Variogram(self.coordinates, self.values, dm=skgstat.tests.binning.binify_dm)

        assert_array_almost_equal(np.asarray(var.bm),
                                  skgstat.tests.binning.result_binify_even_width_bm)

    def test_Variogram_binify_even_width_bm_maxlag(self):
        """
        Testing binning matrix of binify_even_width Function with a maxlag
        """

        var = Variogram(self.coordinates, self.values, maxlag=4)

        assert_array_almost_equal(np.asarray(var.bm),
                                  skgstat.tests.binning.result_binify_even_width_bm_maxlag)

    def test_Variogram_binify_even_width_bw_maxlag(self):
        """
        Testing binning width array of binify_even_width Function with a maxlag
        """

        var = Variogram(self.coordinates, self.values, maxlag=4)
        var.set_bm()

        assert_array_almost_equal(var.bin_widths,
                                  skgstat.tests.binning.result_binify_even_width_bw_maxlag)

    def test_Variogram_binify_even_bin_bm(self):
        """
        Testing binning matrix of binify_even_bin Function with Variogram Class
        """

        var = Variogram(self.coordinates, self.values, bm_func=binify_even_bin)

        assert_array_almost_equal(np.asarray(var.bm),
                                  skgstat.tests.binning.result_binify_even_bin_bm)

    def test_Variogram_binify_even_bin_bw(self):
        """
        Testing binning width array of binify_even_bin Function with Variogram Class
        """

        var = Variogram(self.coordinates, self.values, bm_func=binify_even_bin)
        var.set_bm()

        assert_array_almost_equal(np.asarray(var.bin_widths),
                                  skgstat.tests.binning.result_binify_even_bin_bw)

    def test_Variogram_binify_even_bin_bm_n6(self):
        """
        Testing binning matrix of binify_even_bin Function and 6 bins with Variogram Class
        """

        var = Variogram(self.coordinates, self.values, bm_func=binify_even_bin, N=6)

        assert_array_almost_equal(np.asarray(var.bm),
                                  skgstat.tests.binning.result_binify_even_bin_bm_n6)

    def test_Variogram_binify_even_bin_bw_n6(self):
        """
        Testing binning width array of binify_even_bin Function and 6 bins with Variogram Class
        """

        var = Variogram(self.coordinates, self.values, bm_func=binify_even_bin, N=6)

        assert_array_almost_equal(np.asarray(var.bin_widths),
                                  skgstat.tests.binning.result_binify_even_bin_bw_n6)

    def test_Variogram_binify_even_bin_bm_maxlag(self):
        """
        Testing binning matrix of binify_even_bin Function with a maxlag with Variogram Class
        """

        var = Variogram(self.coordinates, self.values, bm_func=binify_even_bin, maxlag=4)

        assert_array_almost_equal(np.asarray(var.bm),
                                  skgstat.tests.binning.result_binify_even_bin_bm_maxlag)

    def test_Variogram_binify_even_bin_bw_maxlag(self):
        """
        Testing binning width array of binify_even_bin Function with a maxlag with Variogram Class
        """

        var = Variogram(self.coordinates, self.values, bm_func=binify_even_bin, maxlag=4)
        var.set_bm()

        assert_array_almost_equal(np.asarray(var.bin_widths),
                                  skgstat.tests.binning.result_binify_even_bin_bw_maxlag)

    def test_Variogram_binify_group_to_bin(self):
        """
        Testing group_to_bin function with Variogram Class
        """

        var = Variogram(self.coordinates, self.values)

        self.assertEqual(var.grouped_pairs,
                         skgstat.tests.binning.result_group_to_bin)

    def test_Variogram_binify_group_to_bin_with_bm(self):
        """
        Testing group_to_bin function with Variogram Class with bm
        """

        var = Variogram(self.coordinates, self.values)
        var.set_bm(skgstat.tests.binning.group_to_bin_bm)

        self.assertEqual(var.grouped_pairs,
                         skgstat.tests.binning.result_group_to_bin)

    def test_Variogram_binify_group_to_bin_with_maxlag(self):
        """
        Testing group_to_bin function with maxlag in Variogram Class
        """

        var = Variogram(self.coordinates, self.values, maxlag=4)

        self.assertEqual(var.grouped_pairs,
                         skgstat.tests.binning.result_group_to_bin_maxlag)

    def test_Variogram_binify_group_to_bin_with_azimuth_and_tolerance(self):
        """
        Testing group_to_bin function with azimuth and tolerance
        """

        var = Variogram(self.coordinates, self.values, azimuth=90, tolerance=22.5, is_directional=True)

        self.assertEqual(var.grouped_pairs,
                         skgstat.tests.binning.result_group_to_bin_azimuth_and_tolerance)

if __name__ == '__main__':
    unittest.main()
