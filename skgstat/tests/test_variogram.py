import unittest
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from scipy.spatial.distance import pdist

try:
    import plotly.graph_objects as go
    PLOTLY_FOUND = True
except ImportError:
    print('No plotly installed. Skip plot tests')
    PLOTLY_FOUND = False

from skgstat import Variogram, DirectionalVariogram
from skgstat import OrdinaryKriging
from skgstat import estimators
from skgstat import plotting


class TestSpatiallyCorrelatedData(unittest.TestCase):
    def setUp(self):
        # Generate some random but spatially correlated data
        # with a range of ~20
        
        np.random.seed(42)
        c = np.random.sample((50, 2)) * 60
        np.random.seed(42)
        v = np.random.normal(10, 4, 50)
        
        V = Variogram(c, v).describe()
        V["effective_range"] = 20
        OK = OrdinaryKriging(V, coordinates=c, values=v)

        self.c = np.random.sample((500, 2)) * 60
        self.v = OK.transform(self.c)

        self.c = self.c[~np.isnan(self.v),:]
        self.v = self.v[~np.isnan(self.v)]

    def test_dense_maxlag_inf(self):
        Vdense = Variogram(self.c, self.v)
        Vsparse = Variogram(self.c, self.v, maxlag=10000000)

        for x, y in zip(Vdense.parameters, Vsparse.parameters):
            self.assertAlmostEqual(x, y, places=3)
            
    def test_sparse_maxlag_50(self):
        V = Variogram(self.c, self.v, maxlag=50)

        for x, y in zip(V.parameters, [20.264, 6.478, 0]):
            self.assertAlmostEqual(x, y, places=3)
            
    def test_sparse_maxlag_30(self):
        V = Variogram(self.c, self.v, maxlag=30)

        for x, y in zip(V.parameters, [17.128, 6.068, 0]):
            self.assertAlmostEqual(x, y, places=3)


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

    def test_sparse_standard_settings(self):
        V = Variogram(self.c, self.v, maxlag=10000)

        for x, y in zip(V.parameters, [7.122, 13.966, 0]):
            self.assertAlmostEqual(x, y, places=3)

    def test_input_dimensionality(self):
        c1d = np.random.normal(0, 1, 100)
        c3d = np.random.normal(0, 1, size=(100, 3))
        v = np.random.normal(10, 4, 100)

        # test 1D coords
        V = Variogram(c1d, v)
        self.assertTrue(V.dim == 1)

        # test 3D coords
        V2 = Variogram(c3d, v)
        self.assertTrue(V2.dim == 3)

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
            "'notafunc' is not a valid estimator for `bins`",
            str(e.exception)
        )

    def test_invalid_binning_func(self):
        with self.assertRaises(AttributeError) as e:
            V = Variogram(self.c, self.v)
            V.set_bin_func(42)

        self.assertTrue('of type string' in str(e.exception))

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
        with self.assertWarns(Warning) as w:
            Variogram(self.c, [42] * 30, fit_method='lm')

        self.assertEqual(
            'All input values are the same.',
            str(w.warning)
        )

    def test_value_error_on_set_trf(self):
        """Test the Attribute error when switching to TRF on single value input"""
        # catch the same input value warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertRaises(AttributeError) as e:
                v = Variogram(self.c, [42] * 30, fit_method='lm')
                v.fit_method = 'trf'

        self.assertTrue("'trf' is bounded and therefore" in str(e.exception))
    
    def test_value_error_trf(self):
        """Test the Attribute error on TRF instantiation on single value input"""
        # catch the same input value warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertRaises(AttributeError) as e:
                v = Variogram(self.c, [42] * 30, fit_method='trf')

        self.assertTrue("'trf' is bounded and therefore" in str(e.exception))

    def test_pairwise_diffs(self):
        """
        Test that the cross-variogram changes do not mess with the standard
        implementation of Variogram.

        """
        # build the variogram
        V = Variogram(self.c, self.v)

        # build the actual triangular distance matrix array
        diff = pdist(np.column_stack((self.v, np.zeros(len(self.v)))), metric='euclidean')

        assert_array_almost_equal(V.pairwise_diffs, diff, decimal=2)
    
    def test_pairwise_diffs_preprocessing(self):
        """
        Remove the diffs and then request the diffs again to check preprocessing
        trigger on missing pairwise residual diffs.
        """
        V = Variogram(self.c, self.v)

        # build the diffs
        diff = pdist(np.column_stack((self.v, np.zeros(len(self.v)))), metric='euclidean')

        # remove the diffs
        V._diff = None

        # check preprocessing
        assert_array_almost_equal(V.pairwise_diffs, diff, decimal=2)


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

    def test_binning_method_scott(self):
        V = Variogram(self.c, self.v, bin_func='scott')

        # scott should yield 11 bins here
        self.assertTrue(V.n_lags == 11)

        assert_array_almost_equal(
            V.bins,
            np.array([4.9, 8.6, 12.4, 16.1, 19.9, 23.6, 27.3, 31.1, 34.8, 38.6, 42.3]),
            decimal=1
        )

    def test_binning_method_stable(self):
        V = Variogram(self.c, self.v, bin_func='stable_entropy')

        assert_array_almost_equal(
            V.bins,
            np.array([4.3, 8.4, 12.8, 17.1, 21.4, 25.2, 29.9, 33.2, 38.5, 42.8]),
            decimal=1
        )

    def test_binning_method_stable_maxiter(self):
        # increase maxiter - the result should stay the same
        V = Variogram(self.c, self.v, bin_func='stable_entropy', binning_maxiter=20000)

        assert_array_almost_equal(
            V.bins,
            np.array([4.3, 8.4, 12.8, 17.1, 21.4, 25.2, 29.9, 33.2, 38.5, 42.8]),
            decimal=1
        )

    def test_binning_method_stable_fix_bins(self):
        # use 50 bins over the sqrt method - this should change the bins
        V = Variogram(
            self.c,
            self.v,
            bin_func='stable_entropy',
            binning_entropy_bins=50
        )

        assert_array_almost_equal(
            V.bins,
            np.array([4.2, 8.6, 12.8, 17.1, 21.2, 25.5, 29.3, 33.2, 37.4, 43.]),
            decimal=1
        )

    def test_binning_change_nlags(self):
        V = Variogram(self.c, self.v, n_lags=5)

        # 5 lags are awaited
        self.assertTrue(V.n_lags == 5)

        # switch to fd rule
        V.bin_func = 'fd'

        self.assertTrue(V.n_lags == 13)

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

    def test_binning_callable_arg(self):

        # define a custom function similar an existing string function
        def even_func(distances, n, maxlag):
            return np.linspace(0, np.min(np.nanmax(distances), maxlag), n + 1)[1:], None

        # run custom function and string function
        V = Variogram(self.c, self.v, n_lags=8, bin_func=even_func)
        V2 = Variogram(self.c, self.v, n_lags=8, bin_func='even')

        # check the binning is indeed the same
        assert np.array_equal(V.bins, V2.bins)

    def test_binning_iterable_arg(self):

        # define a custom iterable with bin edges
        custom_bins = np.linspace(5,50,5)

        # check that the bins are set according to those edges
        V = Variogram(self.c, self.v, bin_func=custom_bins)

        assert np.array_equal(V.bins, custom_bins)
        assert V.n_lags == len(custom_bins)
        assert V.maxlag == max(custom_bins)

        # check that custom bins have priority over nlags and maxlag
        V = Variogram(self.c, self.v, bin_func=custom_bins, nlags=1000)

        assert np.array_equal(V.bins, custom_bins)
        assert V.n_lags == len(custom_bins)
        assert V.maxlag == max(custom_bins)

        V = Variogram(self.c, self.v, bin_func=custom_bins, maxlag=1000)

        assert np.array_equal(V.bins, custom_bins)
        assert V.n_lags == len(custom_bins)
        assert V.maxlag == max(custom_bins)

    def test_binning_kmeans_method(self):
        V = Variogram(
            self.c,
            self.v,
            n_lags=6,
            bin_func='kmeans',
            binning_random_state=1306
        )

        assert_array_almost_equal(
            V.bins,
            np.array([2.5, 7.7, 12.9, 18.1, 23.7, 30.3]),
            decimal=1
        )

    def test_binning_ward_method(self):
        V = Variogram(self.c, self.v, n_lags=6, bin_func='ward')

        assert_array_almost_equal(
            V.bins,
            np.array([2.5,  7.1, 11.1, 16.2, 23., 30.]),
            decimal=1
        )


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
        # The covariance cannot be estimated here - ignore the warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
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
        """Test to pass a callable as dist function, which always returns 1"""
        # The covariance cannot be estimated here - ignore the warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            V = Variogram([(0, 0), (4, 1), (1, 1)], [1, 2, 3], n_lags=2)

        def dfunc(u, v):
            return 1

        V.set_dist_function(dfunc)

        # test
        self.assertEqual(V.dist_function, dfunc)
        self.assertTrue((V.distance==1).all())
        self.assertEqual(V.distance_matrix.shape, (3, 3))

    @staticmethod
    def disabled_test_direct_dist_setting():
        # Distance can no longer be explicitly set
        # it would require setting the whole MetricSpace, with a
        # non-sparse diagonal matrix
        
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
        self.assertAlmostEqual(
            V.describe()['normalized_nugget'],
            291.28,
            places=2
        )

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
        """Test the distance matrix property for correct shape"""
        coor = [[0, 0], [1, 0], [0, 1], [1, 1]]
        vals = [0, 1, 2, 3]
        dist_mat = np.asarray([
            [0, 1, 1, 1.414],
            [1, 0, 1.414, 1],
            [1, 1.414, 0, 1],
            [1.414, 1, 1, 0]
        ])

        # The covariance cannot be estimated here - ignore the warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
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
            [2.97, 3.3, 3.45, 2.95, 3.33, 3.28, 3.31, 3.44, 2.65, 1.01],
            decimal=2
        )

    def test_metric_space_property(self):
        """
        Test that the MetricSpace is correctly returned
        """
        V = Variogram(self.c, self.v)

        # get the metric space through property
        mc = V.metric_space

        # assert the coords are actually the same
        assert_array_almost_equal(
            mc.coords,
            V.coordinates,
            decimal=5
        )

    def test_metric_space_readonly(self):
        """
        Verify that metric_space is a read-only property.
        """
        V = Variogram(self.c, self.v)

        with self.assertRaises(AttributeError) as e:
            V.metric_space = self.c

            self.assertTrue('read-only' in str(e.exception))

    def test_nofit(self):
        """
        Verify that providing no fitting method skips the fitting procedure
        """
        V = Variogram(self.c, self.v, fit_method=None)

        assert V.fit_method is None
        assert V.cov is None
        assert V.cof is None

    def test_get_bin_count(self):

        V = Variogram(self.c, self.v)

        # check type
        assert isinstance(V.bin_count, np.ndarray)

        # check against real bin count
        assert np.array_equal(V.bin_count, np.array([22, 54, 87, 65, 77, 47, 46, 24, 10,  2]))

        # check property gets updated
        old_bin_count = V.bin_count

        # when setting binning function
        V.bin_func = 'uniform'
        assert not np.array_equal(V.bin_count, old_bin_count)

        # when setting maxlag
        old_bin_count = V.bin_count
        V.maxlag = 25
        assert not np.array_equal(V.bin_count, old_bin_count)

        # when setting nlags
        V.n_lags = 5
        assert len(V.bin_count) == 5


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

    def test_fit_sigma_raises_AttributeError(self):
        self.V.fit_sigma = (0, 1, 2)

        with self.assertRaises(AttributeError) as e:
            self.V.fit_sigma
        
        self.assertTrue(
            'len(fit_sigma)' in str(e.exception)
        )

    def test_fit_sigma_raises_ValueError(self):
        self.V.fit_sigma = 'notAnFunction'

        with self.assertRaises(ValueError) as e:
            self.V.fit_sigma

        self.assertTrue(
            "fit_sigma is not understood." in str(e.exception)
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
            self.V.parameters, [13., 0.3, 18.], decimal=1
        )

    def test_fit_sigma_exp(self):
        self.V.fit_sigma = 'exp'

        # test the sigmas
        sigma = self.V.fit_sigma
        for s, _s in zip(sigma, [0.0067, 0.0821, 0.1889, 0.2865, 0.3679]):
            self.assertAlmostEqual(s, _s, places=4)

        # test parameters
        assert_array_almost_equal(
            self.V.parameters, [25., 0.2, 18.5], decimal=1
        )

    def test_fit_sigma_sqrt(self):
        self.V.fit_sigma = 'sqrt'

        # test the sigmas
        assert_array_almost_equal(
            self.V.fit_sigma, [0.447, 0.632, 0.775, 0.894, 1.], decimal=3
        )

        # test the parameters
        assert_array_almost_equal(
            self.V.parameters, [19.7, 1.5,  16.4], decimal=1
        )

    def test_fit_sigma_sq(self):
        self.V.fit_sigma = 'sq'

        # test the sigmas
        assert_array_almost_equal(
            self.V.fit_sigma, [0.04, 0.16, 0.36, 0.64, 1.], decimal=2
        )

        # test the parameters
        assert_array_almost_equal(
            self.V.parameters, [5.4, 0.1,  18.5], decimal=1
        )

    def test_fit_sigma_entropy(self):
        # load data sample
        data = pd.read_csv(os.path.dirname(__file__) + '/sample.csv')
        V = Variogram(
            data[['x', 'y']].values,
            data.z.values,
            n_lags=12,
            fit_method='ml',
            fit_sigma='entropy'
        )

        assert_array_almost_equal(
            V.parameters, [65.9, 1.3, 0], decimal=1
        )

    def test_fit_sigma_on_the_fly(self):
        self.V.fit(sigma='sq')

        # test the sigmas
        assert_array_almost_equal(
            self.V.fit_sigma, [0.04, 0.16, 0.36, 0.64, 1.], decimal=2
        )

        # test the parameters
        assert_array_almost_equal(
            self.V.parameters, [5.4, 0.1,  18.5], decimal=1
        )

    def test_fit_lm(self):
        df = pd.read_csv(os.path.dirname(__file__) + '/sample.csv')
        V = Variogram(
            df[['x', 'y']],
            df.z.values,
            use_nugget=True,
            n_lags=8, fit_method='lm'
        )

        # test the parameters
        assert_array_almost_equal(
            V.parameters, [162.3, 0.5, 0.8], decimal=1
        )

    def test_fitted_model(self):
        self.V._fit_method = 'trf'
        self.V.fit_sigma = None
        fun = self.V.fitted_model

        result = np.array([12.48, 17.2, 17.2, 17.2])

        assert_array_almost_equal(
            result, list(map(fun, np.arange(0, 20, 5))),
            decimal=2
        )

    def test_unavailable_method(self):
        with self.assertRaises(AttributeError) as e:
            self.V.fit(method='unsupported')

        self.assertTrue(
            "fit_method has to be one of" in str(e.exception)
        )

    def test_implicit_run_fit_fitted_model(self):
        self.V.fit_sigma = None
        self.V._fit_method = 'trf'
        result = np.array([12.48, 17.2, 17.2, 17.2])

        # remove cof
        self.V.cof = None

        # test on fitted model
        fun = self.V.fitted_model

        assert_array_almost_equal(
            result, list(map(fun, np.arange(0, 20, 5))), decimal=2
        )

    def test_implicit_run_fit_transform(self):
        self.V.fit_sigma = None
        self.V._fit_method = 'trf'
        result = np.array([12.48, 17.2, 17.2, 17.2])

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

    def test_ml_default(self):
        # load data sample
        df = pd.read_csv(os.path.dirname(__file__) + '/sample.csv')
        V = Variogram(
            df[['x', 'y']],
            df.z.values,
            use_nugget=True,
            n_lags=15,
            fit_method='ml'
        )

        assert_array_almost_equal(
            V.parameters, np.array([41.18, 1.2, 0.]), decimal=2
        )

    def test_ml_sq_sigma(self):
        # load data sample
        df = pd.read_csv(os.path.dirname(__file__) + '/sample.csv')
        V = Variogram(
            df[['x', 'y']],
            df.z.values,
            use_nugget=True,
            n_lags=15,
            fit_method='ml',
            fit_sigma='sq'
        )

        assert_array_almost_equal(
            V.parameters, np.array([42.72, 1.21, 0.]), decimal=2
        )

    def test_manual_fit(self):
        V = Variogram(
            self.c,
            self.v,
            fit_method='manual',
            model='spherical',
            fit_range=10.,
            fit_sill=5.
        )

        self.assertEqual(V.parameters, [10., 5., 0.0])
    
    def test_manual_fit_change(self):
        V = Variogram(
            self.c,
            self.v,
            fit_method='trf',
            model='matern',
        )

        # switch to manual fit
        V._fit_method = 'manual'
        V.fit(range=10, sill=5, shape=3)

        self.assertEqual(V.parameters, [10., 5., 3., 0.0])

    def test_manual_raises_missing_params(self):
        with self.assertRaises(AttributeError) as e:
            Variogram(self.c, self.v, fit_method='manual')
            self.assertTrue('For manual fitting' in str(e.exception))

    def test_manual_preserve_params(self):
        V = Variogram(self.c, self.v, fit_method='trf', n_lags=8)
        params = V.parameters

        # switch fit method
        V._fit_method = 'manual'
        V.fit(sill=14)

        # expected output
        params[1] = 14.

        assert_array_almost_equal(
            V.parameters,
            params,
            decimal=1
        )
    
    def test_implicit_nugget(self):
        V = Variogram(self.c, self.v, use_nugget=False)

        # no nugget used
        self.assertTrue(V.parameters[-1] < 1e-10)

        # switch to manual fitting
        V.fit(method='manual', sill=5., nugget=2.)

        self.assertTrue(abs(V.parameters[-1] - 2.) < 1e-10)


class TestVariogramQualityMeasures(unittest.TestCase):
    def setUp(self):
        # set up default values, whenever c and v are not important
        np.random.seed(42)
        self.c = np.random.gamma(10, 4, (30, 2))
        np.random.seed(42)
        self.v = np.random.normal(10, 4, 30)

    def test_residuals(self):
        V = Variogram(self.c, self.v)
        assert_array_almost_equal(
            V.model_residuals,
            np.array(
                [-3.43e-08, -1.33e-01, 2.11e+00, 4.89e+00, 1.37e+00, 1.50e+00,
                 -3.83e+00, -6.89e+00, 3.54e+00, -2.55e+00]),
            decimal=2
        )

    def test_rmse(self):
        V = Variogram(self.c, self.v)

        for model, rmse in zip(
                ['spherical', 'gaussian', 'stable'],
                [3.3705, 3.3707, 3.193]
        ):
            V.set_model(model)
            self.assertAlmostEqual(V.rmse, rmse, places=4)

    def test_mean_residual(self):
        V = Variogram(self.c, self.v)

        for model, mr in zip(
            ['spherical', 'cubic', 'stable'],
            [2.6803, 2.6803, 2.6966]
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
        
    def test_mae(self):
        V = Variogram(self.c, self.v, n_lags=15)

        self.assertAlmostEqual(V.mae, 3.91, places=2)

    def test_mse(self):
        V = Variogram(self.c, self.v, n_lags=15)

        self.assertAlmostEqual(np.sqrt(V.mse), V.rmse, places=6)

    def test_update_kwargs(self):
        V = Variogram(self.c, self.v, percentile=.3)

        self.assertAlmostEqual(
            V._kwargs.get('percentile'), 0.3, places=1
        )

        # change the parameter
        V.update_kwargs(percentile=0.7)

        self.assertAlmostEqual(
            V._kwargs.get('percentile'), 0.7, places=1
        )

    def test_kwargs_setter_in_experimental(self):
        V = Variogram(self.c, self.v, estimator='percentile')

        # store with p of 50 == median
        exp = V.experimental

        V.update_kwargs(percentile=25)

        exp2 = V.experimental

        # 25% should be very different from median
        with self.assertRaises(AssertionError):
            assert_array_almost_equal(exp, exp2, decimal=2)

    def test_residuals_deprecation(self):
        """Variogram.residuals is deprecated in favor of model_residuals"""
        with self.assertWarns(DeprecationWarning) as w:
            Variogram(self.c, self.v).residuals

        self.assertTrue('residuals is deprecated and will be removed' in str(w.warning))


class TestVariogramMethods(unittest.TestCase):
    def setUp(self):
        # set up default values, whenever c and v are not important
        np.random.seed(42)
        self.c = np.random.gamma(10, 4, (30, 2))
        np.random.seed(42)
        self.v = np.random.normal(10, 4, 30)

        self.V = Variogram(self.c, self.v, normalize=False, n_lags=10)

    def test_get_empirical(self):
        bins = self.V.bins
        exp = self.V.experimental

        emp_x, emp_y = self.V.get_empirical()

        # test
        assert_array_almost_equal(bins, emp_x)
        assert_array_almost_equal(exp, emp_y)
    
    def test_get_empirical_center(self):
        V = Variogram(self.c, self.v)

        # overwrite bins
        V.bins = [4, 8, 9, 12, 15]
        emp_x, emp_y = V.get_empirical(bin_center=True)

        assert_array_almost_equal(emp_x, [2., 6., 8.5, 10.5, 13.5])

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

    def disabled_test_data_with_force(self):
        # Distance can no longer be explicitly set
        # it would require setting the whole MetricSpace, with a
        # non-sparse diagonal matrix
        
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
        param = [42.3, 16.2, 0.1, 0.]
        V.set_model('matern')
        assert_array_almost_equal(V.parameters, param, decimal=2)
    
    def test_parameter_property_stable(self):
        V = self.V.clone()

        # test stable
        param = [42.3, 15.79, 0.45,  0.]
        V.set_model('stable')
        assert_array_almost_equal(V.parameters, param, decimal=2)


class TestVariogramPlotlyPlots(unittest.TestCase):
    def setUp(self):
        # set up default values, whenever c and v are not important
        np.random.seed(42)
        self.c = np.random.gamma(10, 4, (150, 2))
        np.random.seed(42)
        self.v = np.random.normal(10, 4, 150)
        self.V = Variogram(self.c, self.v)

    def test_plotly_main_plot(self):
        if PLOTLY_FOUND:
            # switch to plotly
            plotting.backend('plotly')

            self.assertTrue(
                isinstance(self.V.plot(show=False), go.Figure)
            )

            plotting.backend('matplotlib')

    def test_plotly_scattergram(self):
        if PLOTLY_FOUND:
            # switch to plotly
            plotting.backend('plotly')

            self.assertTrue(
                isinstance(self.V.scattergram(show=False), go.Figure)
            )

            plotting.backend('matplotlib')

    def test_plotly_location_trend(self):
        if PLOTLY_FOUND:
            # switch to plotly
            plotting.backend('plotly')

            self.assertTrue(
                isinstance(self.V.location_trend(show=False), go.Figure)
            )

            plotting.backend('matplotlib')

    def test_plotly_dd_plot(self):
        if PLOTLY_FOUND:
            # switch to plotly
            plotting.backend('plotly')

            self.assertTrue(
                isinstance(self.V.distance_difference_plot(show=False), go.Figure)
            )

            plotting.backend('matplotlib')
    
    def test_undefined_backend(self):
        # force the backend into an undefined state
        import skgstat
        skgstat.__backend__ = 'not-a-backend'

        for fname in ('plot', 'scattergram', 'location_trend', 'distance_difference_plot'):
            with self.assertRaises(ValueError) as e:
                self.V.plot()

                self.assertEqual(
                    str(e.exception),
                    'The plotting backend has an undefined state.'
                )
        
        # make the backend valid again
        skgstat.__backend__ = 'matplotlib'


class TestSampling(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(os.path.dirname(__file__) + '/pan_sample.csv')

    def test_full_vs_full_sample(self):
        Vf = Variogram(
            self.data[['x', 'y']].values,
            self.data.z.values,
            binning_random_state=44).describe()

        Vs = Variogram(
            self.data[['x', 'y']].values,
            self.data.z.values, samples=len(self.data),
            binning_random_state=44).describe()

        self.assertAlmostEqual(Vf["normalized_effective_range"], Vs["normalized_effective_range"], delta = Vf["normalized_effective_range"] / 10)
        self.assertAlmostEqual(Vf["effective_range"], Vs["effective_range"], delta = Vf["effective_range"] / 10)
        self.assertAlmostEqual(Vf["sill"], Vs["sill"], delta = Vf["sill"] / 10)

    def test_samples(self):
        Vf = Variogram(
            self.data[['x', 'y']].values,
            self.data.z.values, samples=len(self.data),
            binning_random_state=44).describe()

        for sample_size in np.linspace(0.5, 1., 10):
            Vs = Variogram(
                self.data[['x', 'y']].values,
                self.data.z.values, samples=sample_size,
                binning_random_state=44).describe()
        
            self.assertAlmostEqual(Vf["normalized_effective_range"], Vs["normalized_effective_range"], delta = Vf["normalized_effective_range"] / 5)
            self.assertAlmostEqual(Vf["effective_range"], Vs["effective_range"], delta = Vf["effective_range"] / 5)
            self.assertAlmostEqual(Vf["sill"], Vs["sill"], delta = Vf["sill"] / 5)


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
        return True


class TestCrossVariogram(unittest.TestCase):
    def setUp(self):
        # ignore scipy runtime warnings as for this random data
        # the covariance may not be positive-semidefinite
        # this is caused by the multivariate_normal - thus no problem
        # see here: https://stackoverflow.com/questions/41515522/numpy-positive-semi-definite-warning
        warnings.simplefilter('ignore', category=RuntimeWarning)

        # set up default values, whenever c and v are not important
        np.random.seed(42)
        self.c = np.random.gamma(10, 4, (100, 2))
        np.random.seed(42)
        self.v = np.random.multivariate_normal([10.5, 5.6], [[1.5, 3.2], [3.2, 1.5]], size=100)

    def test_cross_variable_dim_error(self):
        """Generate values of too high dimensionality"""
        # generate 3D values
        np.random.seed(42)
        v = np.random.normal(10, 4, (100, 3, 3))

        with self.assertRaises(ValueError) as e:
            Variogram(self.c, v)

        self.assertTrue('values has to be 1d (classic variogram)' in str(e.exception))

    def test_too_many_cross_variables(self):
        """Generate too many co-variables"""
        np.random.seed(42)
        v = np.random.normal(10, 3, (100, 3))

        with self.assertRaises(ValueError) as e:
            Variogram(self.c, v)

        self.assertTrue('create a grid of cross-variograms' in str(e.exception))

    def test_cross_instantiation(self):
        """Create a cross-variogram without error"""
        vario = Variogram(self.c, self.v, maxlag='median')

        self.assertTrue(vario.is_cross_variogram)

    def test_cross_shapes(self):
        """Check that the cross-variogram does not change the diff shape"""
        vario = Variogram(self.c, self.v, maxlag='median')

        self.assertTrue(vario.pairwise_diffs.ndim == 1)

    def test_covariable(self):
        """check that the covariable was correctly instantiated"""
        vario = Variogram(self.c, self.v, maxlag='median')

        self.assertTrue(vario.is_cross_variogram)
        assert_array_almost_equal(self.v[:,1], vario._co_variable)

    def test_directional_instantiation(self):
        """Check that the directional variogram is also instantiated."""
        vario = DirectionalVariogram(self.c, self.v, maxlag='median')

        self.assertTrue(vario.pairwise_diffs.ndim == 1)

    def test_directional_covariable(self):
        """Check that the directional variogram instantiated the covariable correctly"""
        vario = DirectionalVariogram(self.c, self.v, maxlag='median')

        self.assertTrue(vario.is_cross_variogram)
        assert_array_almost_equal(self.v[:,1], vario._co_variable)

    def test_cross_variogram_warns(self):
        """Test warning when cross-variogram is exported to gstools"""
        vario = Variogram(self.c, self.v)

        with self.assertWarns(Warning) as w:
            vario.to_gstools()
        
        self.assertTrue("This instance is a cross-variogram!!" in str(w.warning))


if __name__ == '__main__':  # pragma: no cover
    os.environ['SKG_SUPRESS'] = 'TRUE'
    unittest.main()
