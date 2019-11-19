import unittest

import numpy as np 
from numpy.testing import assert_array_almost_equal
from sklearn.model_selection import GridSearchCV

from skgstat.interfaces import VariogramEstimator


class TestVariogramEstimator(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.c = np.random.normal(50, 8, (300, 2))
        np.random.seed(42)
        self.v = np.random.gamma(10, 1, 300)

    def test_default(self):
        """
        Test default instantition by comparing the fitted 
        bins and experimental variogram values to expected arrays.

        """
        # test the built Variogram against bins and experimental
        ve = VariogramEstimator(n_lags=8, normalize=False)

        # fit, to obtain Variogram
        ve_fit = ve.fit(self.c, self.v)
        V = ve_fit.variogram

        assert_array_almost_equal(
            V.bins, 
            [6.67, 13.35, 20.02, 26.7, 33.37, 40.04, 46.72, 53.39], 
            decimal=2
        )

        assert_array_almost_equal(
            V.experimental,
            [11.01, 10.47, 9.84, 9.17, 8.5, 6.64, 7.41, 4.54],
            decimal=2
        )

    def test_predict_func(self):
        """
        Test predict function. should return the same thing as the 
        transform function of the underlying variogram.

        """
        ve = VariogramEstimator(n_lags=15, normalize=False).fit(self.c, self.v)
        v = ve.variogram

        x = np.linspace(0, ve.range_, 100)

        assert_array_almost_equal(ve.predict(x), v.transform(x), decimal=6)

    def test_find_best_model(self):
        """
        Use GridSearchCV to find the best model for the given data
        which should be the spherical model
        """
        parameters = dict(
            model=('spherical', 'gaussian', 'exponential', 'matern')
        )
        gs = GridSearchCV(VariogramEstimator(n_lags=15, normalize=False), parameters)

        gs = gs.fit(self.c, self.v)

        self.assertEqual(gs.best_params_['model'], 'spherical')


if __name__=='__main__':
    import os
    os.environ['SKG_SUPRESS'] = 'TRUE' # pragma: no cover
    unittest.main() # pragma: no cover
