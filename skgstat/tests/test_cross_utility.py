import unittest
import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal

from skgstat import Variogram, DirectionalVariogram
from skgstat.util.cross_variogram import cross_variograms


class TestCrossUtility(unittest.TestCase):
    def setUp(self) -> None:
        # ignore scipy runtime warnings as for this random data
        # the covariance may not be positive-semidefinite
        # this is caused by the multivariate_normal - thus no problem
        # see here: https://stackoverflow.com/questions/41515522/numpy-positive-semi-definite-warning
        warnings.simplefilter('ignore', category=RuntimeWarning)

        # set up default values, whenever c and v are not important
        np.random.seed(42)
        self.c = np.random.gamma(10, 4, (100, 2))
        
        # build the multivariate sample
        means = [1, 10, 100, 1000]
        cov = [[1, 0.8, 0.7, 0.6], [0.8, 1, 0.2, 0.2], [0.7, 0.2, 1.0, 0.2], [0.6, 0.2, 0.2, 1.0]]
        
        np.random.seed(42)
        self.v = np.random.multivariate_normal(means, cov, size=100)

    def test_cross_matrix_shape(self):
        """Test the shape of the cross-variogram matrix for 4 variables"""
        mat = cross_variograms(self.c, self.v)

        # check shape
        mat = np.asarray(mat, dtype='object')
        self.assertTrue(mat.shape, (4, 4))
    
    def test_cross_matrix_diagonal(self):
        """Test that the primary variograms are correct"""
        # get the cross variogram matrix
        mat = cross_variograms(self.c, self.v, maxlag='median')

        # calculate the first and third primary variogram
        first = Variogram(self.c, self.v[:, 0], maxlag='median')
        third = Variogram(self.c, self.v[:, 2], maxlag='median')

        # assert first empirical variogram
        assert_array_almost_equal(mat[0][0].experimental, first.experimental, 2)
        assert_array_almost_equal(mat[0][0].bins, first.bins, 1)

        # assert thrird empirical variogram
        assert_array_almost_equal(mat[2][2].experimental, third.experimental, 2)
        assert_array_almost_equal(mat[2][2].bins, third.bins, 1)

    def test_check_cross_variogram(self):
        """Test two of the cross-variograms in the matrix"""
        mat = cross_variograms(self.c, self.v, n_lags=15)

        # calculate two cross-variograms
        first = Variogram(self.c, self.v[:, [1, 3]], n_lags=15)
        second = Variogram(self.c, self.v[:, [0, 2]], n_lags=15)

        # assert first variogram
        assert_array_almost_equal(mat[1][3].experimental, first.experimental, 2)
        assert_array_almost_equal(mat[1][3].bins, first.bins, 1)

        # assert second variogram
        assert_array_almost_equal(mat[0][2].experimental, second.experimental, 2)
        assert_array_almost_equal(mat[0][2].bins, second.bins, 1)

    def test_for_directional_variograms(self):
        """Check that DirectionalVariograms are also calcualted correctly"""
        mat = cross_variograms(self.c, self.v, azimuth=90)

        mat = np.asarray(mat, dtype='object').flatten()

        self.assertTrue(all([isinstance(v, DirectionalVariogram) for v in mat]))
