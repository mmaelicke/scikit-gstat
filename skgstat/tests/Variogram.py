import unittest
import os
import pickle

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt

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

        for x, y in zip(V.parameters, [7.122, 13.966, 0]):
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
    
    def test_unknown_binning_func(self):
        with self.assertRaises(ValueError) as e:
            Variogram(self.c, self.v, bin_func='notafunc')

        self.assertEqual(
            'notafunc binning method is not known',
            str(e.exception)
        )

    def test_unknown_model(self):
        with self.assertRaises(ValueError) as e:
            Variogram(self.c, self.v, model='unknown')

        self.assertEqual(
            'The theoretical Variogram function unknown is not understood, please provide the function',
            str(e.exception)
        )

    def test_unsupported_n_lags(self):
        with self.assertRaises(ValueError) as e:
            Variogram(self.c, self.v, n_lags=15.7)

        self.assertEqual(
            'n_lags has to be a positive integer',
            str(e.exception)
        )

    def test_value_warning(self):
        with self.assertRaises(Warning) as w:
            Variogram(self.c, [42] * 30)
        
        self.assertEqual(
            'All input values are the same.',
            str(w.exception)
        )


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
            str(e.exception),
            'The estimator has to be a string or callable.'
        )

    def test_set_unknown_estimator(self):
        V = Variogram(self.c, self.v)

        with self.assertRaises(ValueError) as e:
            V.set_estimator('notaestimator')

        self.assertEqual(
            str(e.exception),
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
            str(e.exception),
            'Unknown Distance Metric: notadistance'
        )

    def test_wrong_dist_func_input(self):
        V = Variogram(self.c, self.v)

        with self.assertRaises(ValueError) as e:
            V.set_dist_function(55)
            
        self.assertEqual(
            str(e.exception),
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

    def test_maxlag_custom_value(self):
        V = Variogram(self.c, self.v)

        V.maxlag = 33.3
        self.assertAlmostEqual(V.maxlag, 33.3, places=1)

    def test_use_nugget_setting(self):
        V = Variogram(self.c, self.v, normalize=True)

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
            str(e.exception),
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
                str(e.exception),
                'n_lags has to be a positive integer'
            )

    def test_n_lags_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            Variogram(self.c, self.v, n_lags='auto')
    
    def test_set_values(self):
        V = Variogram(self.c, self.v)

        # create a new array of same length
        _old_vals = V.values
        new_vals = np.random.normal(10, 2, size=len(_old_vals))

        V.values = new_vals

        # values.setter will call set_values
        assert_array_almost_equal(V.values, new_vals, decimal=4)

    def test_value_matrix(self):
        vals = np.array([1, 2, 3, 4])
        mat = np.asarray([[0, 1, 2, 3], [1, 0, 1, 2],[2, 1, 0, 1], [3, 2, 1, 0]], dtype=int)

        V = Variogram(self.c[:4], vals)

        assert_array_almost_equal(V.value_matrix, mat, decimal=1)

    def _test_normalize_setter(self):
        # TODO: I should fix this behavior
        V = Variogram(self.c, self.v, normalize=False)

        # make sure biggest bin larger than 1.0
        self.assertGreater(np.max(V.bins), 1.0)

        # normalize
        V.normalize = True

        # now, biggest bin should be almost or exactly 1.0
        self.assertLessEqual(np.max(V.bins), 1.0)
    
    def test_distance_matrix(self):
        coor = [[0, 0], [1, 0], [0, 1], [1, 1]]
        vals = [0, 1, 2, 3]
        dist_mat = np.asarray([
            [0, 1, 1, 1.414],
            [1, 0, 1.414, 1],
            [1, 1.414, 0, 1],
            [1.414, 1, 1, 0]
        ])

        V = Variogram(coor, vals)

        assert_array_almost_equal(V.distance_matrix, dist_mat, decimal=3)
    
    def test_entropy_as_estimator(self):
        """
        Note: This unittest will change in future, as soon as the 
        bin edges for Entropy calculation can be set on instantiation

        """
        V = Variogram(self.c, self.v, estimator='entropy', n_lags=10)

        assert_array_almost_equal(
            V.experimental, 
            [2.97, 3.3 , 3.45, 2.95, 3.33, 3.28, 3.31, 3.44, 2.65, 1.01],
            decimal=2
        )


class TestVariogramFittingProcedure(unittest.TestCase):
    def setUp(self):
        np.random.seed(1337)
        self.c = np.random.gamma(10, 8, (50, 3))
        np.random.seed(1337)
        self.v = np.random.normal(10, 4, 50)

        # build a standard variogram to be used
        self.V = Variogram(
            self.c, self.v, n_lags=5, normalize=False, use_nugget=True
        )

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
            [24.008, 17.083, 0.99], decimal=3
        )

    def test_fit_sigma_raises_AttributeError(self):
        self.V.fit_sigma = (0, 1, 2)

        with self.assertRaises(AttributeError) as e:
            self.V.fit_sigma
        
        self.assertEqual(
            str(e.exception),
            'fit_sigma and bins need the same length.'
        )

    def test_fit_sigma_raises_ValueError(self):
        self.V.fit_sigma = 'notAnFunction'

        with self.assertRaises(ValueError) as e:
            self.V.fit_sigma
            
        self.assertEqual(
            str(e.exception),
            "fit_sigma is not understood. It has to be an array or one of ['linear', 'exp', 'sqrt', 'sq']."
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
            self.V.parameters, [25.077, 17.393, 0.925], decimal=3
        )

    def test_fit_sigma_exp(self):
        self.V.fit_sigma = 'exp'

        # test the sigmas
        sigma = self.V.fit_sigma
        for s, _s in zip(sigma, [0.0067, 0.0821, 0.1889, 0.2865, 0.3679]):
            self.assertAlmostEqual(s, _s, places=4)

        # test parameters
        assert_array_almost_equal(
            self.V.parameters, [25.3, 17.7, 1.], decimal=1
        )

    def test_fit_sigma_sqrt(self):
        self.V.fit_sigma = 'sqrt'

        # test the sigmas
        assert_array_almost_equal(
            self.V.fit_sigma, [0.447, 0.632, 0.775, 0.894, 1.], decimal=3
        )

        # test the parameters
        assert_array_almost_equal(
            self.V.parameters, [23., 17.,  1.], decimal=1
        )

    def test_fit_sigma_sq(self):
        self.V.fit_sigma = 'sq'

        # test the sigmas
        assert_array_almost_equal(
            self.V.fit_sigma, [0.04, 0.16, 0.36, 0.64, 1.], decimal=2
        )

        # test the parameters
        assert_array_almost_equal(
            self.V.parameters, [25.3, 17.6,  1.], decimal=1
        )
    
    def test_fit_sigma_on_the_fly(self):
        self.V.fit(sigma='sq')

        # test the sigmas
        assert_array_almost_equal(
            self.V.fit_sigma, [0.04, 0.16, 0.36, 0.64, 1.], decimal=2
        )

        # test the parameters
        assert_array_almost_equal(
            self.V.parameters, [25.3, 17.6,  1.], decimal=1
        )

    def test_fit_lm(self):
        self.V.fit(method='lm', sigma='sqrt')

        # test the parameters
        assert_array_almost_equal(
            self.V.parameters, [1., 17., 1.], decimal=0
        )

    def test_fitted_model(self):
        fun = self.V.fitted_model

        result = [0.99, 7.19, 12.53, 16.14]

        assert_array_almost_equal(
            result, list(map(fun, np.arange(0, 20, 5))),
            decimal=2
        )

    def test_unavailable_method(self):
        with self.assertRaises(ValueError) as e:
            self.V.fit(method='unsupported')

        self.assertEqual(
            "fit method has to be one of ['trf', 'lm']",
            str(e.exception)
        )
  
    def test_implicit_run_fit_fitted_model(self):
        self.V.fit_sigma = None
        self.V.fit_method = 'trf'
        result = [0.99,  7.19, 12.53, 16.14]

        # remove cof
        self.V.cof = None

        # test on fitted model
        fun = self.V.fitted_model

        assert_array_almost_equal(
            result, list(map(fun, np.arange(0, 20, 5))), decimal=2
        )

    def test_implicit_run_fit_transform(self):
        self.V.fit_sigma = None
        self.V.fit_method = 'trf'
        result = [0.99,  7.19, 12.53, 16.14]

        # test on transform
        self.V.cof = None
        res = self.V.transform(np.arange(0, 20, 5))

        assert_array_almost_equal(result, res, decimal=2)

    def test_harmonize_model(self):
        # load data sample
        data = pd.read_csv(os.path.dirname(__file__) + '/sample.csv')
        V = Variogram(data[['x', 'y']].values, data.z.values)

        V.model = 'harmonize'
        x = np.linspace(0, np.max(V.bins), 10)

        assert_array_almost_equal(
            V.transform(x),
            [np.NaN, 0.57, 1.01, 1.12, 1.15, 1.15, 1.15, 1.15, 1.21, 1.65],
            decimal=2
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

    def test_r(self):
        V = Variogram(self.c, self.v, n_lags=12, normalize=False)

        for model, r in zip(
            ('gaussian', 'exponential', 'stable'), 
            [0.39, 0.55, 0.60]
        ):
            V.set_model(model)
            self.assertAlmostEqual(V.r, r, places=2)
    
    def test_NS(self):
        V = Variogram(self.c, self.v, n_lags=15, normalize=False)

        for estimator, NS in zip(
            ('matheron', 'genton', 'dowd'),
            [0.0206, 0.0206, 0.0206]
        ):
            self.assertAlmostEqual(V.NS, NS, places=4)


class TestVariogramMethods(unittest.TestCase):
    def setUp(self):
        # set up default values, whenever c and v are not important
        np.random.seed(42)
        self.c = np.random.gamma(10, 4, (30, 2))
        np.random.seed(42)
        self.v = np.random.normal(10, 4, 30)

        self.V = Variogram(self.c, self.v, normalize=False, n_lags=10)

    def test_clone_method(self):
        # copy variogram
        copy = self.V.clone()

        # test against bins and experimental
        assert_array_almost_equal(copy.experimental, self.V.experimental)
        assert_array_almost_equal(copy.bins, self.V.bins)

    def test_data_no_force(self):
        lags, var = self.V.data(n=10, force=False)

        assert_array_almost_equal(
            lags,
            [0.,  4.7,  9.4, 14.1, 18.8, 23.5, 28.2, 32.9, 37.6, 42.3], 
            decimal=2
        )

        assert_array_almost_equal(
            var,
            [0., 11.82, 13.97, 13.97, 13.97, 13.97, 13.97, 13.97, 13.97, 13.97],
            decimal=2
        )
    
    def test_data_with_force(self):
        # should work if _dist is corccupted
        self.V._dist = self.V._dist * 5.
        self.V.cof = None
        lags, var = self.V.data(n=10, force=True)

        assert_array_almost_equal(
            lags,
            [0., 4.7, 9.4, 14.1, 18.8, 23.5, 28.2, 32.9, 37.6, 42.3],
            decimal=2
        )

        assert_array_almost_equal(
            var,
            [0., 11.82, 13.97, 13.97, 13.97, 13.97, 13.97, 13.97, 13.97, 13.97],
            decimal=2
        )

    def test_data_normalized(self):
        V = self.V.clone()

        V.normalize = True

        lags, var = V.data(n=5, force=True)

        assert_array_almost_equal(
            lags,
            [0., 10.58, 21.15, 31.73, 42.3],
            decimal=2
        )

        assert_array_almost_equal(
            var,
            [0., 13.97, 13.97, 13.97, 13.97],
            decimal=2
        )
    
    def test_parameter_property_matern(self):
        V = self.V.clone()
        
        # test matern
        param = [42.3, 16.2,  0.1,  0.]
        V.set_model('matern')
        assert_array_almost_equal(V.parameters, param, decimal=2)
    
    def test_parameter_property_stable(self):
        V = self.V.clone()

        # test stable
        param = [42.3 , 15.79, 0.45,  0.]
        V.set_model('stable')
        assert_array_almost_equal(V.parameters, param, decimal=2)


class TestVariogramPlots(unittest.TestCase):
    def setUp(self):
        # set up default values, whenever c and v are not important
        np.random.seed(42)
        self.c = np.random.gamma(10, 4, (150, 2))
        np.random.seed(42)
        self.v = np.random.normal(10, 4, 150)

    def test_main_plot(self):
        V = Variogram(self.c, self.v, n_lags=5, normalize=True)

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

    def test_main_plot_pass_axes(self):
        V = Variogram(self.c, self.v, n_lags=5, normalize=True)

        # build the figure externally
        fig, axes = plt.subplots(1, 2)
        fig = V.plot(axes=axes, show=False)
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

    def test_main_plot_not_normalized(self):
        V = Variogram(self.c, self.v, n_lags=5, normalize=False)

        # build the figure
        fig = V.plot(show=False)
        ax1, ax2 = fig.axes

        # test experimental
        assert_array_almost_equal(
            [12.7 , 15., 17.98, 15.9, 15.39],
            ax1.get_children()[1].get_data()[1],
            decimal=2
        )

        #  test theoretical at some locations
        assert_array_almost_equal(
            [ 2.9 , 10.18, 15.86, 16.07],
            ax1.get_children()[2].get_data()[1][[4, 15, 30, 50]],
            decimal=2
        )       

    def test_main_plot_histogram(self):
        V = Variogram(self.c, self.v, n_lags=5, normalize=True)

        # build the figure
        fig = V.plot(show=False)
        childs = fig.axes[1].get_children()

        # test histogram
        for i, h in zip(range(1, 6), [5262, 4674, 1047, 142, 49]):
            self.assertEqual(childs[i].get_height(), h)

    def test_main_plot_no_histogram(self, normalize=True):
        V = Variogram(self.c, self.v, n_lags=5)

        # two axes
        fig = V.plot(show=False)
        self.assertEqual(len(fig.axes), 2)

        fig = V.plot(hist=False, show=False)
        self.assertEqual(len(fig.axes), 1)

    def test_default_scattergram(self):
        V = Variogram(self.c, self.v, n_lags=5)

        fig = V.scattergram(show=False)
        ax = fig.axes[0]

        # test just 3 positions of the scatterplot
        assert_array_almost_equal(
            [[12., 9.1], [2.3, 13.3], [13.1, 9.4]],
            ax.get_children()[2].get_offsets()[[5, 1117, 523]],
            decimal=True
        )

    def test_scattergram_on_ax(self):
        V = Variogram(self.c, self.v, n_lags=5)

        # define figure
        fig, ax = plt.subplots(1,1)
        V.scattergram(show=False, ax=ax)

        # test just 3 positions of the scatterplot
        assert_array_almost_equal(
            [[12., 9.1], [2.3, 13.3], [13.1, 9.4]],
            ax.get_children()[2].get_offsets()[[5, 1117, 523]],
            decimal=True
        )
    
    def test_location_trend(self):
        # test that the correct amount of axes is produced
        V = Variogram(self.c, self.v)

        fig = V.location_trend(show=False)

        self.assertEqual(len(fig.axes), 2)

    def test_location_trend_pass_axes(self):
        V = Variogram(self.c, self.v)

        fig, axes = plt.subplots(1, 2)
        V.location_trend(axes=axes, show=False)

        # test some random y-values
        assert_array_almost_equal(
            [9.06, 5.95, 4.34, 10.91],
            axes[0].get_children()[0].get_data()[1][[4, 16, 100, 140]],
            decimal=2
        )

    def test_location_trend_raises(self):
        V = Variogram(self.c, self.v)

        with self.assertRaises(ValueError):
            V.location_trend(axes=[0, 1, 2])

    def test_distance_difference_plot(self):
        V = Variogram(self.c, self.v, n_lags=4)

        fig = V.distance_difference_plot(show=False)
        ax = fig.axes[0]
        
        # test some scatter positions
        assert_array_almost_equal(
            [[14.77, 4.33], [30.11, 7.87], [11.32, 2.63]],
            ax.get_children()[1].get_offsets()[[5, 112, 1337]],
            decimal=2
        )

    def test_distance_difference_pass_ax(self):
        V = Variogram(self.c, self.v, n_lags=4)

        fig, ax = plt.subplots(1,1)
        V.distance_difference_plot(ax=ax, show=False)

        # test some scatter positions
        assert_array_almost_equal(
            [[14.77, 4.33], [30.11, 7.87], [11.32, 2.63]],
            ax.get_children()[1].get_offsets()[[5, 112, 1337]],
            decimal=2
        )

    def test_distance_difference_with_bins(self):
        V = Variogram(self.c, self.v, n_lags=4)

        fig1 = V.distance_difference_plot(show=False, plot_bins=False)
        fig2 = V.distance_difference_plot(show=False, plot_bins=True)

        # there should be one child less
        self.assertEqual(
            len(fig1.axes[0].get_children()),
            len(fig2.axes[0].get_children()) - 1
        )


class TestVariogramPickling(unittest.TestCase):
    def setUp(self):
        # set up default values, whenever c and v are not important
        np.random.seed(42)
        self.c = np.random.gamma(10, 4, (150, 2))
        np.random.seed(42)
        self.v = np.random.normal(10, 4, 150)

        self.V = Variogram(self.c, self.v, normalize=False)

    def test_save_load_pickle(self):
        """
        Only test if loading and saving a pickle works without error
        """
        pickle.loads(pickle.dumps(self.V))
        self.assertTrue()


if __name__ == '__main__':  # pragma: no cover
    import os
    os.environ['SKG_SUPRESS'] = 'TRUE'
    unittest.main()
