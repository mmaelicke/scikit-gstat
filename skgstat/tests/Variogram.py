import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from skgstat import Variogram
from skgstat import estimators


class TestVariogramInstatiation(unittest.TestCase):
    def setUp(self):
        # set up default values, whenever c and v are not important
        np.random.seed(42)
        self.c = np.random.gamma(10, 4, (30, 2))
        np.random.seed(42)
        self.v = np.random.normal(10, 4, 30)

    def test_standard_settings(self):
        V = Variogram(self.c, self.v)

        for x, y in zip(V.parameters, [301.266, 291.284, 0]):
            self.assertAlmostEqual(x, y, places=3)

    def test_pass_median_maxlag_on_instantiation(self):
        np.random.seed(1312)
        c = np.random.gamma(5, 1, (50, 2))

        np.random.seed(1312)
        v = np.random.weibull(5, 50)

        V = Variogram(c, v, maxlag='median', n_lags=4)
        bins = [0.88, 1.77, 2.65, 3.53]

        for b, e in zip(bins, V.bins):
            self.assertAlmostEqual(b, e, places=2)

    def test_pass_mean_maxlag_on_instantiation(self):
        V = Variogram(self.c, self.v, maxlag='mean', n_lags=4)

        bins = [4.23, 8.46, 12.69, 16.91]

        for b, e in zip(bins, V.bins):
            self.assertAlmostEqual(b, e, places=2)


class TestVariogramArguments(unittest.TestCase):
    def setUp(self):
        # set up default values, whenever c and v are not important
        np.random.seed(42)
        self.c = np.random.gamma(10, 4, (30, 2))
        np.random.seed(42)
        self.v = np.random.normal(10, 4, 30)

    def test_binning_method_setting(self):
        V = Variogram(self.c, self.v, n_lags=4)

        # lags
        even = [10.58, 21.15, 31.73, 42.3]
        uniform = [10.25, 16.21, 22.71, 42.3]

        # test even
        assert_array_almost_equal(even, V.bins, decimal=2)

        # set to uniform
        V.set_bin_func('uniform')
        assert_array_almost_equal(uniform, V.bins, decimal=2)

        # restore even
        V.bin_func = 'even'
        assert_array_almost_equal(even, V.bins, decimal=2)

    def test_set_bins_directly(self):
        V = Variogram(self.c, self.v, n_lags=5)

        # set bins by hand
        bins = np.array([4., 20., 21., 25., 40.])
        V.bins = bins

        # test setting
        assert_array_almost_equal(bins, V.bins, decimal=8)

        # test cov settings
        self.assertIsNone(V.cov)
        self.assertIsNone(V.cof)

    def test_estimator_method_setting(self):
        """
        Only test if the estimator functions are correctly set. The
        estimator functions themselves are tested in a unittest of their own.
        """
        V = Variogram(self.c, self.v, n_lags=4)

        estimator_list = ('cressie', 'matheron', 'dowd', 'genton', 'minmax',
                      'percentile', 'entropy')

        for estimator in estimator_list:
            # set the estimator
            V.estimator = estimator
            imported_estimator = getattr(estimators, estimator)
            self.assertEqual(imported_estimator, V.estimator)

    def test_set_estimator_wrong_type(self):
        V = Variogram(self.c, self.v)

        with self.assertRaises(ValueError) as e:
            V.set_estimator(45)
            self.assertEqual(
                str(e),
                'The estimator has to be a string or callable.'
            )

    def test_set_unknown_estimator(self):
        V = Variogram(self.c, self.v)

        with self.assertRaises(ValueError) as e:
            V.set_estimator('notaestimator')
            self.assertEqual(
                str(e),
                'Variogram estimator notaestimator is not understood, please ' +
                'provide the function.'
            )

    def test_set_dist_func(self):
        V = Variogram([(0, 0), (4, 1), (1, 1)], [1, 2, 3], n_lags=2)

        # use Manhattan distance
        V.set_dist_function('cityblock')
        for d, v in zip([5., 2., 3.], V.distance):
            self.assertEqual(d, v)

    def test_unknown_dist_func(self):
        V = Variogram(self.c, self.v)

        with self.assertRaises(ValueError) as e:
            V.set_dist_function('notadistance')
            self.assertEqual(
                str(e),
                'Unknown Distance Metri: notadistance'
            )

    def test_wrong_dist_func_input(self):
        V = Variogram(self.c, self.v)

        with self.assertRaises(ValueError) as e:
            V.set_dist_function(55)
            self.assertEqual(
                str(e),
                'Input not supported. Pass a string or callable.'
            )

    def test_callable_dist_function(self):
        V = Variogram([(0, 0), (4, 1), (1, 1)], [1, 2, 3], n_lags=2)

        def dfunc(x):
            return np.ones((len(x), len(x)))

        V.set_dist_function(dfunc)

        # test
        self.assertEqual(V.dist_function, dfunc)
        self.assertTrue((V.distance==1).all())
        self.assertEqual(V.distance.shape, (3, 3))

    @staticmethod
    def test_direct_dist_setting():
        V = Variogram([(0, 0), (4, 1), (1, 1)], [1, 2, 3], n_lags=2)

        V.distance = np.array([0, 0, 100])

        assert_array_almost_equal(V.distance, [0, 0, 100], decimal=0)

    def test_maxlag_setting_as_max_ratio(self):
        V = Variogram(self.c, self.v)

        # set maxlag to 60% of maximum distance
        V.maxlag = 0.6
        self.assertEqual(V.maxlag, np.max(V.distance) * 0.6)
        self.assertAlmostEqual(V.maxlag, 25.38, places=2)

    def test_use_nugget_setting(self):
        V = Variogram(self.c, self.v)

        # test the property and setter
        self.assertEqual(V.use_nugget, False)
        self.assertEqual(V.describe()['nugget'], 0)

        # set the nugget
        V.use_nugget = True
        self.assertEqual(V.use_nugget, True)
        self.assertEqual(V._use_nugget, True)
        self.assertAlmostEqual(V.describe()['nugget'], 291.28, places=2)

    def test_use_nugget_exception(self):
        with self.assertRaises(ValueError) as e:
            Variogram(self.c, self.v, use_nugget=42)
            self.assertEqual(
                str(e),
                'use_nugget has to be of type bool.'
            )

    def test_n_lags_change(self):
        V = Variogram(self.c, self.v, n_lags=10)

        self.assertEqual(len(V.bins), 10)
        V.n_lags = 5
        self.assertEqual(len(V.bins), 5)

    def test_n_lags_exception(self):
        for arg in [15.5, -5]:
            with self.assertRaises(ValueError) as e:
                Variogram(self.c, self.v, n_lags=arg)
                self.assertEqual(
                    str(e),
                    'n_lags has to be a positive integer'
                )

    def test_n_lags_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            Variogram(self.c, self.v, n_lags='auto')


class TestVariogramFittingProcedure(unittest.TestCase):
    def setUp(self):
        np.random.seed(1337)
        self.c = np.random.gamma(10, 8, (50, 3))
        np.random.seed(1337)
        self.v = np.random.normal(10, 4, 50)

        # build a standard variogram to be used
        self.V = Variogram(self.c, self.v, n_lags=5, use_nugget=True)

    def test_fit_sigma_is_None(self):
        self.V.fit_sigma = None

        self.assertIsNone(self.V.fit_sigma)

    def test_fit_sigma_explicit(self):
        sigs = [.8, .5, 2., 2., 5.]
        self.V.fit_sigma = sigs

        for x, y in zip(sigs, self.V.fit_sigma):
            self.assertEqual(x, y)

        # test parameter estimated
        self.V.fit()
        assert_array_almost_equal(
            self.V.parameters,
            [3035.357, 318.608, 18.464], decimal=3
        )

    def test_fit_sigma_raises_AttributeError(self):
        self.V.fit_sigma = (0, 1, 2)

        with self.assertRaises(AttributeError) as e:
            self.V.fit_sigma
            self.assertEqual(
                str(e),
                'fit_sigma and bins need the same length.'
            )

    def test_fit_sigma_raises_ValueError(self):
        self.V.fit_sigma = 'notAnFunction'

        with self.assertRaises(ValueError) as e:
            self.V.fit_sigma
            self.assertEqual(
                str(e),
                "fit_sigma is not understood. It has to be an array or" +
                "one of ['linear', 'exp', 'sqrt', 'sq']."
            )

    def test_fit_sigma_linear(self):
        self.V.fit_sigma = 'linear'

        # test the sigmas
        sigma = self.V.fit_sigma
        for s, _s in zip(sigma, [.2, .4, .6, .8, 1.]):
            self.assertAlmostEqual(s, _s, places=8)

        # test parameters:
        self.V.fit()
        assert_array_almost_equal(
            self.V.parameters, [3170.532, 324.385, 17.247], decimal=3
        )

    def test_fit_sigma_exp(self):
        self.V.fit_sigma = 'exp'

        # test the sigmas
        sigma = self.V.fit_sigma
        for s, _s in zip(sigma, [0.0067, 0.0821, 0.1889, 0.2865, 0.3679]):
            self.assertAlmostEqual(s, _s, places=4)

        # test parameters
        assert_array_almost_equal(
            self.V.parameters, [3195.6, 329.8, 17.9], decimal=1
        )

    def test_fit_sigma_sqrt(self):
        self.V.fit_sigma = 'sqrt'

        # test the sigmas
        assert_array_almost_equal(
            self.V.fit_sigma, [0.447, 0.632, 0.775, 0.894, 1.], decimal=3
        )

        # test the parameters
        assert_array_almost_equal(
            self.V.parameters, [2902., 315., 18.], decimal=0
        )

    def test_fit_sigma_sq(self):
        self.V.fit_sigma = 'sq'

        # test the sigmas
        assert_array_almost_equal(
            self.V.fit_sigma, [0.04, 0.16, 0.36, 0.64, 1.], decimal=2
        )

        # test the parameters
        assert_array_almost_equal(
            self.V.parameters, [3195., 328.9, 17.8], decimal=1
        )


class TestVariogramQaulityMeasures(unittest.TestCase):
    def setUp(self):
        # set up default values, whenever c and v are not important
        np.random.seed(42)
        self.c = np.random.gamma(10, 4, (30, 2))
        np.random.seed(42)
        self.v = np.random.normal(10, 4, 30)

    def test_residuals(self):
        V = Variogram(self.c, self.v)
        assert_array_almost_equal(
            V.residuals,
            np.array(
                [-3.43e-08, -1.33e-01, 2.11e+00, 4.89e+00, 1.37e+00, 1.50e+00,
                 -3.83e+00, -6.89e+00, 3.54e+00, -2.55e+00]),
            decimal=2
        )

    def test_rmse(self):
        V = Variogram(self.c, self.v)

        for model, rmse in zip(
                ['spherical', 'gaussian', 'matern', 'stable'],
                [3.3705, 3.3707, 3.1548, 3.193]
        ):
            V.set_model(model)
            self.assertAlmostEqual(V.rmse, rmse, places=4)

    def test_mean_residual(self):
        V = Variogram(self.c, self.v)

        for model, mr in zip(
            ['spherical', 'cubic', 'matern', 'stable'],
            [2.6803, 2.6803, 2.637, 2.6966]
        ):
            V.set_model(model)
            self.assertAlmostEqual(V.mean_residual, mr, places=4)

    def test_nrmse(self):
        V = Variogram(self.c, self.v, n_lags=15)

        for model, nrmse in zip(
            ['spherical', 'gaussian', 'stable', 'exponential'],
            [0.3536, 0.3535, 0.3361, 0.3499]
        ):
            V.set_model(model)
            self.assertAlmostEqual(V.nrmse, nrmse, places=4)

    def test_nrmse_r(self):
        V = Variogram(self.c, self.v, estimator='cressie')

        self.assertAlmostEqual(V.nrmse_r, 0.63543, places=5)


class TestVariogramPlots(unittest.TestCase):
    def setUp(self):
        # set up default values, whenever c and v are not important
        np.random.seed(42)
        self.c = np.random.gamma(10, 4, (150, 2))
        np.random.seed(42)
        self.v = np.random.normal(10, 4, 150)

    def test_main_plot(self):
        V = Variogram(self.c, self.v, n_lags=5)

        # build the figure
        fig = V.plot(show=False)
        ax1, ax2 = fig.axes

        # test experimental
        assert_array_almost_equal(
            [0.71, 0.83, 1., 0.88, 0.86],
            ax1.get_children()[1].get_data()[1],
            decimal=2
        )

        #  test theoretical at some locations
        assert_array_almost_equal(
            [0.16, 0.57, 0.88, 0.89],
            ax1.get_children()[2].get_data()[1][[4, 15, 30, 50]],
            decimal=2
        )

    def test_main_plot_histogram(self):
        V = Variogram(self.c, self.v, n_lags=5)

        # build the figure
        fig = V.plot(show=False)
        childs = fig.axes[1].get_children()

        # test histogram
        for i, h in zip(range(1, 6), [5262, 4674, 1047, 142, 49]):
            self.assertEqual(childs[i].get_height(), h)

    def test_main_plot_no_histogram(self):
        V = Variogram(self.c, self.v, n_lags=5)

        # two axes
        fig = V.plot(show=False)
        self.assertEqual(len(fig.axes), 2)

        fig = V.plot(hist=False, show=False)
        self.assertEqual(len(fig.axes), 1)


if __name__ == '__main__':
    unittest.main()
