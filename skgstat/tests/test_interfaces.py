import os
import sys
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from sklearn.model_selection import GridSearchCV

from skgstat import Variogram, OrdinaryKriging
from skgstat.interfaces import VariogramEstimator


def get_sample() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'pan_sample.csv'))
    return df


try:
    import pykrige
    print(f'Found PyKrige: {pykrige.__version__}')
    from skgstat.interfaces import pykrige as pykrige_interface
    PYKRIGE_AVAILABLE = True
except ImportError:
    PYKRIGE_AVAILABLE = False  # pragma: no cover

try:
    import gstools
    print(f'Found PyKrige: {gstools.__version__}')
    GSTOOLS_AVAILABLE = True
except ImportError:
    GSTOOLS_AVAILABLE = False  # pragma: no cover


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
        gs = GridSearchCV(
            VariogramEstimator(n_lags=15, normalize=False),
            parameters,
            cv=3
        )

        gs = gs.fit(self.c, self.v)

        # Python 3.6 yields 'exponential', 
        # while 3.7, 3.8 yield 'gaussian' - this is so stupid
        self.assertTrue(gs.best_params_['model'] in ['gaussian', 'exponential'])

    def test_find_best_model_future_cv(self):
        """
        cv parameter will change to 5 in sklearn 0.22
        This will change the result, though
        """
        parameters = dict(
            model=('spherical', 'gaussian', 'exponential', 'matern')
        )
        gs = GridSearchCV(
            VariogramEstimator(n_lags=15, normalize=False),
            parameters,
            cv=5
        )

        gs = gs.fit(self.c, self.v)

        self.assertEqual(gs.best_params_['model'], 'exponential')

    def test_cross_validation_option(self):
        # this test does not support python < 3.8
        # the code works, but GridSearchCV gives dififerent results
        if sys.version_info[1] < 8:
            return True

        df = get_sample()
        c = df[['x', 'y']].values
        v = df.z.values

        parameters = dict(
            bin_func=('even', 'scott', 'sqrt')
        )

        # based on model fit
        gs_nocv = GridSearchCV(
            VariogramEstimator(model='exponential', maxlag=100, n_lags=25, cross_n=50),
            parameters,
            cv=3
        )
        gs_nocv = gs_nocv.fit(c, v)
        self.assertEqual(gs_nocv.best_params_['bin_func'], 'sqrt')

        # based on cross-validation
        gs = GridSearchCV(
            VariogramEstimator(
                model='exponential',
                maxlag=100,
                n_lags=25,
                cross_n=50,
                cross_validate=True,
                use_score='mae',
                seed=42
                ),
            parameters,
            cv=3
        )
        gs = gs.fit(c, v)
        return True


class TestPyKrigeInterface(unittest.TestCase):
    def setUp(self):
        # use real sample data in the interface
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'sample.csv'))
        self.c = df[['x', 'y']].values
        self.v = df.z.values

        self.V = Variogram(self.c, self.v, model='matern', normalize=False, use_nugget=True)

        if not PYKRIGE_AVAILABLE:
            print('PyKrige not found, will skip all pykrige interface tests')

    def test_model_interface(self):
        if not PYKRIGE_AVAILABLE:  # pragma: no cover
            return True
        # get the function
        model = pykrige_interface.pykrige_model(self.V)

        # use the transform function.
        xi = np.arange(1, 85)
        yi = self.V.transform(xi)

        assert_array_almost_equal(yi, model([], xi), decimal=6)

    def test_model_interface_from_list(self):
        if not PYKRIGE_AVAILABLE:  # pragma: no cover
            return True

        # get the function
        model = pykrige_interface.pykrige_model(self.V)

        # use the transform function
        xi = list(range(1, 85))
        yi = self.V.transform(np.array(xi))

        assert_array_almost_equal(yi, model([], xi), decimal=6)

    def test_parameters(self):
        if not PYKRIGE_AVAILABLE:  # pragma: no cover
            return True

        p = pykrige_interface.pykrige_params(self.V)
        params = self.V.parameters

        self.assertAlmostEqual(p[0], params[1], places=4)
        self.assertAlmostEqual(p[1], params[0], places=4)
        self.assertAlmostEqual(p[2], params[2], places=4)

    def test_as_kwargs(self):
        if not PYKRIGE_AVAILABLE:  # pragma: no cover
            return True

        args = pykrige_interface.pykrige_as_kwargs(self.V)
        pars = pykrige_interface.pykrige_params(self.V)

        # test
        self.assertEqual(args['variogram_model'], 'custom')
        assert_array_almost_equal(pars, args['variogram_parameters'])

        xi = np.arange(1, 80)
        yi = self.V.transform(xi)
        assert_array_almost_equal(
            yi,
            args['variogram_function']([], xi),
            decimal=6
        )

    def test_as_kwargs_adjust_maxlag(self):
        if not PYKRIGE_AVAILABLE:  # pragma: no cover
            return True

        V = self.V.clone()

        # now maxlag should be changed
        args = pykrige_interface.pykrige_as_kwargs(V, adjust_maxlag=True)

        # should be None
        self.assertIsNone(V.maxlag)

        # transform should change
        xi = np.arange(1, 20)
        yi = V.transform(xi)

        # test changed values
        assert_array_almost_equal(yi, args['variogram_function']([], xi))

    def test_as_kwargs_adjust_nlags(self):
        if not PYKRIGE_AVAILABLE:  # pragma: no cover
            return True

        args = pykrige_interface.pykrige_as_kwargs(self.V, adjust_nlags=True)

        self.assertEqual(args['nlags'], self.V.n_lags)


class TestGstoolsInterface(unittest.TestCase):
    def setUp(self):
        # use real sample data in the interface
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'sample.csv'))
        self.c = df[['x', 'y']].values
        self.v = df.z.values

        # build variogram
        self.V = Variogram(self.c, self.v, model='stable', normalize=False, use_nugget=True)

        # model data
        self.xi = np.linspace(0, self.V.bins[-1], 100)
        self.yi = self.V.transform(self.xi)

        if not GSTOOLS_AVAILABLE:
            print('GSTools not found, will skip all gstools interface tests')

    def test_interface(self):
        if not GSTOOLS_AVAILABLE:  # pragma: no cover
            return True

        model = self.V.to_gstools(dim=2)

        assert_array_almost_equal(
            model.variogram(self.xi), self.yi, decimal=2
        )

    def test_infer_dims(self):
        if not GSTOOLS_AVAILABLE:  # pragma: no cover
            return True

        model = self.V.to_gstools()

        assert_array_almost_equal(
            model.variogram(self.xi), self.yi, decimal=2
        )


class TestGstoolsAllModels(unittest.TestCase):
    def setUp(self):
        # use real sample data in the interface
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'sample.csv'))
        self.c = df[['x', 'y']].values
        self.v = df.z.values

        if not GSTOOLS_AVAILABLE:
            print('GSTools not found, will skip all gstools interface tests')

    def assert_model(self, model: str):
        if not GSTOOLS_AVAILABLE:  # pragma: no cover
            return True

        # build the model
        V = Variogram(self.c, self.v, model=model, normalize=False)

        # model data
        xi = np.linspace(0, V.bins[-1], 100)
        yi = V.transform(xi)

        model = V.to_gstools()

        assert_array_almost_equal(
            model.variogram(xi), yi, decimal=2
        )

    def test_stable_model(self):
        self.assert_model('stable')

    def test_spherical_model(self):
        self.assert_model('spherical')

    def test_exponential_model(self):
        self.assert_model('exponential')

    def test_cubic_model(self):
        self.assert_model('cubic')

    def test_gaussian_model(self):
        self.assert_model('gaussian')

    def test_matern_model(self):
        self.assert_model('matern')


class TestGstoolsKrige(unittest.TestCase):
    def setUp(self):
        # use real sample data in the interface
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'sample.csv'))
        self.c = df[['x', 'y']].values
        self.v = df.z.values

        # build variogram
        self.V = Variogram(self.c[1:], self.v[1:], model='stable', normalize=False, use_nugget=False)

    def test_ordinary(self):
        if not GSTOOLS_AVAILABLE:  # pragma: no cover
            return True

        x = np.array([self.c[0][0]])
        y = np.array([self.c[0][1]])

        # run ordinary kriging with skgstat
        ok = OrdinaryKriging(self.V, min_points=3)
        sk_res = ok.transform(x, y)

        # get the gstools Krige class
        krige = self.V.to_gs_krige()
        gs_res, _ = krige.structured([x, y])

        # test
        assert_array_almost_equal(
            sk_res.flatten(),
            gs_res.flatten(),
            decimal=1
        )


if __name__ == '__main__':
    os.environ['SKG_SUPRESS'] = 'TRUE'  # pragma: no cover
    unittest.main()  # pragma: no cover
