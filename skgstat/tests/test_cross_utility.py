import unittest
import warnings

import numpy as np

from skgstat.util.cross_variogram import cross_variogram


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
        mat = cross_variogram(self.c, self.v)

        # check shape
        mat = np.asarray(mat, dtype='object')
        self.assertTrue(mat.shape, (4, 4))