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

        for x, y in zip(V.parameters, [439.405, 281.969, 0]):
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
            np.array([0.96, -1.19, -0.28, 2.61, -0.9,
                      -0.43, -0.1, -2.32, -8.61, 10.61]),
            decimal=2
        )

    def test_rmse(self):
        V = Variogram(self.c, self.v)

        for model, rmse in zip(
                ['spherical', 'gaussian', 'matern', 'stable'],
                [4.4968, 4.4878, 4.4905, 4.4878]
        ):
            V.set_model(model)
            self.assertAlmostEqual(V.rmse, rmse, places=4)

    def test_mean_residual(self):
        V = Variogram(self.c, self.v)

        for model, mr in zip(
            ['spherical', 'cubic', 'matern', 'stable'],
            [2.8006, 2.711, 2.7433, 2.7315]
        ):
            V.set_model(model)
            self.assertAlmostEqual(V.mean_residual, mr, places=4)

    def test_nrmse(self):
        V = Variogram(self.c, self.v, n_lags=15)

        for model, nrmse in zip(
            ['spherical', 'gaussian', 'stable', 'exponential'],
            [0.4751, 0.4784, 0.4621, 0.4695]
        ):
            V.set_model(model)
            self.assertAlmostEqual(V.nrmse, nrmse, places=4)

    def test_nrmse_r(self):
        V = Variogram(self.c, self.v, estimator='cressie')

        self.assertAlmostEqual(V.nrmse_r, 0.40796, places=5)


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
            [0.71, 0.7, 0.81, 1., 0.86],
            ax1.get_children()[1].get_data()[1],
            decimal=2
        )

        #  test theoretical at some locations
        assert_array_almost_equal(
            [0.17, 0.58, 0.84, 0.84],
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
