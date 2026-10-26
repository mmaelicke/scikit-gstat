import unittest
import pickle
import warnings
from datetime import datetime

import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd
import matplotlib.pyplot as plt

from skgstat import SpaceTimeVariogram, SpatioTemporalVariogram, estimators, plotting


class TestSpatioTemporalVariogramInitialization(unittest.TestCase):
    def setUp(self):
        # four observations: two locations at two times
        self.c = np.array([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.]
        ])
        self.v = np.array([0., 10., 100., 1000.])

    def test_uses_all_pairs(self):
        V = SpatioTemporalVariogram(self.c, self.v, x_lags=1, t_lags=1)

        # all 6 pairs fall into the only lag class
        expected = 0.5 * np.mean(np.square([10., 100., 1000., 90., 990., 900.]))
        self.assertEqual(V.bin_count.tolist(), [[6]])
        self.assertAlmostEqual(V.experimental[0, 0], expected)

    def test_marginals_use_zero_lag_pairs(self):
        V = SpatioTemporalVariogram(self.c, self.v, x_lags=1, t_lags=1)

        # space marginal: same time, time marginal: same location
        self.assertAlmostEqual(
            V.get_marginal('space')[0],
            0.5 * np.mean(np.square([100., 990.]))
        )
        self.assertAlmostEqual(
            V.get_marginal('time')[0],
            0.5 * np.mean(np.square([10., 900.]))
        )

    def test_marginal_without_zero_lags_raises(self):
        V = SpatioTemporalVariogram(
            np.array([[0., 0.], [1., 1.], [2., 2.]]), [0., 1., 4.]
        )

        with self.assertRaises(ValueError):
            V.get_marginal('space')

    def test_duplicates_raise(self):
        c = np.array([[10., 20., 0.], [10., 20., 0.], [11., 20., 1.]])

        with self.assertRaises(ValueError) as e:
            SpatioTemporalVariogram(c, [1., 2., 3.])

        self.assertIn('rows [0, 1]', str(e.exception))

    def test_non_finite_raises(self):
        with self.assertRaises(ValueError):
            SpatioTemporalVariogram(self.c, [0., np.nan, 1., 2.])

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            SpatioTemporalVariogram(self.c, [0., 1., 2.])

    def test_single_observation_raises(self):
        with self.assertRaises(ValueError):
            SpatioTemporalVariogram([[0., 0.]], [1.])

    def test_spatial_metric_excludes_time(self):
        V = SpatioTemporalVariogram(
            np.array([[0., 0., 0.], [3., 4., 100.]]),
            [0., 1.],
            xdist_func='cityblock',
            x_lags=1,
            t_lags=1
        )

        self.assertEqual(V.xdistance.tolist(), [7.])
        self.assertEqual(V.tdistance.tolist(), [100.])

    def test_pickle(self):
        V = SpatioTemporalVariogram(self.c, self.v, x_lags=1, t_lags=1)
        restored = pickle.loads(pickle.dumps(V))

        assert_array_almost_equal(restored.experimental, V.experimental)


class TestSpatioTemporalVariogramTime(unittest.TestCase):
    def test_datetime_is_anchored(self):
        c = pd.DataFrame({
            'x': [0., 1., 2.],
            'time': pd.date_range('2024-01-01', periods=3, freq='D')
        })
        V = SpatioTemporalVariogram(c, [1., 2., 4.], x_lags=1, t_lags=2)

        self.assertEqual(V.time_origin, pd.Timestamp('2024-01-01'))
        self.assertEqual(V.time.tolist(), [0., 86400., 172800.])
        self.assertEqual(V.time_unit, 'days')

    def test_datetime_types(self):
        for times in (
            [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            np.array(['2024-01-01', '2024-01-02'], dtype='datetime64[D]'),
            [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')]
        ):
            c = np.empty((2, 2), dtype=object)
            c[:, 0] = [0., 1.]
            c[:, 1] = times
            V = SpatioTemporalVariogram(c, [0., 1.], x_lags=1, t_lags=1)

            self.assertEqual(V.time.tolist(), [0., 86400.])
            self.assertEqual(V.time_unit, 'days')

    def test_numeric_time_has_no_unit(self):
        c = np.column_stack(([0., 1., 2.], [0., 60., 120.]))
        V = SpatioTemporalVariogram(c, [1., 2., 3.])

        self.assertIsNone(V.time_origin)
        self.assertIsNone(V.time_unit)


class TestSpatioTemporalVariogramArguments(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.c = np.random.gamma(10, 6, (15, 2))
        np.random.seed(42)
        self.v = np.random.normal(15, 4, (15, 12))

    def test_maxlag_filters_pairs(self):
        V = SpatioTemporalVariogram(
            np.array([[0., 0.], [1., 1.], [10., 10.]]),
            [0., 1., 2.],
            x_lags=1,
            t_lags=1,
            maxlag_space=2.,
            maxlag_time=2.
        )

        self.assertEqual(V.bin_count.tolist(), [[1]])

    def test_relative_maxlag(self):
        V = SpatioTemporalVariogram.from_grid(self.c, self.v, maxlag_space=0.5, maxlag_time=0.5)

        self.assertAlmostEqual(V.xbins[-1], 0.5 * V.xdistance.max())
        self.assertAlmostEqual(V.tbins[-1], 0.5 * V.tdistance.max())

    def test_median_maxlag(self):
        V = SpatioTemporalVariogram.from_grid(self.c, self.v, maxlag_space='median')

        self.assertAlmostEqual(V.xbins[-1], np.median(V.xdistance))

    def test_explicit_bin_edges(self):
        V = SpatioTemporalVariogram(
            np.array([[0., 0.], [1., 0.], [3., 1.]]),
            [0., 1., 2.],
            xbins=[1., 3.],
            tbins=[0., 1.]
        )

        self.assertEqual(V.xbins.tolist(), [1., 3.])
        self.assertEqual(V.tbins.tolist(), [0., 1.])
        self.assertEqual(V.experimental.shape, (2, 2))

    def test_binning_methods(self):
        for method in ('even', 'uniform', 'kmeans', 'ward', 'sturges'):
            V = SpatioTemporalVariogram.from_grid(self.c, self.v, x_lags=5, t_lags=5, xbins=method)
            self.assertTrue(np.all(np.diff(V.xbins) > 0))

    def test_zero_axis_collapses(self):
        space = SpatioTemporalVariogram(np.array([[0., 0.], [1., 0.], [2., 0.]]), [0., 1., 2.])
        time = SpatioTemporalVariogram(np.array([[0., 0.], [0., 1.], [0., 2.]]), [0., 1., 2.])

        self.assertEqual(space.tbins.tolist(), [0.])
        self.assertEqual(time.xbins.tolist(), [0.])
        self.assertEqual(space.bin_count.sum(), 3)
        self.assertEqual(time.bin_count.sum(), 3)

    def test_estimator(self):
        V = SpatioTemporalVariogram(
            np.array([[0., 0.], [1., 1.], [2., 2.]]),
            [0., 1., 4.],
            estimator='cressie',
            x_lags=1,
            t_lags=1
        )

        self.assertAlmostEqual(V.experimental[0, 0], estimators.cressie(np.array([1., 4., 3.])))

    def test_unknown_estimator_raises(self):
        for name in ('foobar', 'np', 'njit'):
            with self.assertRaises(ValueError):
                SpatioTemporalVariogram.from_grid(self.c, self.v, estimator=name)

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            SpatioTemporalVariogram.from_grid(self.c, self.v, model='batmobile')

    def test_unknown_marginal_model_raises(self):
        for name in ('foobar', 'variogram'):
            with self.assertRaises(ValueError):
                SpatioTemporalVariogram.from_grid(self.c, self.v, marginal_model=name)

    def test_invalid_lags_raise(self):
        with self.assertRaises(ValueError):
            SpatioTemporalVariogram.from_grid(self.c, self.v, x_lags=0)
        with self.assertRaises(ValueError):
            SpatioTemporalVariogram.from_grid(self.c, self.v, t_lags=2.5)


class TestSpatioTemporalVariogramGrid(unittest.TestCase):
    def test_every_cell_is_an_observation(self):
        values = np.array([[0., 1., 2.], [10., 11., 12.]])
        V = SpatioTemporalVariogram.from_grid(
            np.array([[0.], [1.]]), values, times=[0., 1., 2.], x_lags=1, t_lags=1
        )

        self.assertEqual(V.values.tolist(), values.flatten().tolist())
        self.assertEqual(V.bin_count.sum(), 15)

    def test_missing_cells_are_dropped(self):
        values = np.array([[0., np.nan, 2.], [10., 11., 12.]])
        V = SpatioTemporalVariogram.from_grid(np.array([[0.], [1.]]), values, x_lags=1, t_lags=1)

        self.assertEqual(V.values.tolist(), [0., 2., 10., 11., 12.])
        self.assertEqual(V.bin_count.sum(), 10)

    def test_datetime_times(self):
        V = SpatioTemporalVariogram.from_grid(
            np.array([[0.], [1.]]),
            np.array([[0., 1.], [2., 3.]]),
            times=pd.date_range('2024-01-01', periods=2, freq='h')
        )

        self.assertEqual(V.time_unit, 'hours')

    def test_legacy_class_warns(self):
        with self.assertWarns(FutureWarning):
            SpaceTimeVariogram(np.array([[0.], [1.]]), np.array([[0., 1.], [2., 3.]]), x_lags=1, t_lags=1)


class TestSpatioTemporalVariogramFit(unittest.TestCase):
    def setUp(self):
        # additive field: spatial trend plus temporal trend
        self.c = np.arange(5.)[:, None]
        self.t = np.arange(6.)
        self.v = np.add.outer(np.square(self.c[:, 0]), np.sqrt(self.t))

    def test_sum_model(self):
        V = SpatioTemporalVariogram.from_grid(self.c, self.v, times=self.t, x_lags=4, t_lags=5, model='sum')
        z = V.transform(np.array([[0., 0.], [1., 1.]]))

        self.assertEqual(V.cof.size, 0)
        self.assertEqual(z.shape, (2, ))
        self.assertAlmostEqual(z[0], 0.)

    def test_product_model(self):
        V = SpatioTemporalVariogram.from_grid(self.c, self.v, times=self.t, x_lags=4, t_lags=5, model='product')
        V.fit()

        self.assertEqual(set(V._model_params.keys()), {'Cx', 'Ct'})
        self.assertTrue(np.all(np.isfinite(V.transform(np.array([[1., 2.]])))))

    def test_product_sum_model(self):
        V = SpatioTemporalVariogram.from_grid(self.c, self.v, times=self.t, x_lags=4, t_lags=5)
        V.fit()
        z = V.transform(np.array([[1., 2.], [3., 4.]]))

        self.assertEqual(len(V.cof), 3)
        self.assertTrue(np.all(np.isfinite(z)))

    def test_marginal_models(self):
        for name in ('spherical', 'exponential', 'gaussian', 'cubic', 'stable', 'matern'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                V = SpatioTemporalVariogram.from_grid(
                    self.c, self.v, times=self.t, x_lags=4, t_lags=5, marginal_model=name, model='sum'
                )
                z = V.transform(np.array([[1., 2.]]))

            self.assertTrue(np.all(np.isfinite(z)), name)

    def test_nugget(self):
        V = SpatioTemporalVariogram.from_grid(
            self.c, self.v, times=self.t, x_lags=4, t_lags=5, use_nugget=True, model='product'
        )
        V.fit()

        # range, sill, nugget and the nugget is part of the marginal sill
        cof = V._marginal_cof['space']
        self.assertEqual(len(cof), 3)
        self.assertAlmostEqual(V._model_params['Cx'], cof[1] + cof[2])

    def test_fitted_pickle(self):
        V = SpatioTemporalVariogram.from_grid(self.c, self.v, times=self.t, x_lags=4, t_lags=5, model='sum')
        expected = V.transform(np.array([[1., 1.]]))
        restored = pickle.loads(pickle.dumps(V))

        assert_array_almost_equal(restored.transform(np.array([[1., 1.]])), expected)


class TestSpatioTemporalVariogramPlots(unittest.TestCase):
    def setUp(self):
        self.c = pd.DataFrame({
            'x': [0., 0., 1., 1., 2., 2.],
            'time': pd.to_datetime(['2024-01-01', '2024-01-02'] * 3)
        })
        self.V = SpatioTemporalVariogram(self.c, [0., 1., 2., 3., 5., 4.], x_lags=2, t_lags=1)

    def tearDown(self):
        plt.close('all')

    def test_matrix_uses_time_unit(self):
        fig = self.V.plot(kind='matrix')

        self.assertEqual(fig.axes[0].get_ylabel(), 'time lag [days]')
        self.assertAlmostEqual(fig.axes[0].get_ylim()[1], 1.)

    def test_default_is_3d_scatter(self):
        fig = self.V.plot()

        self.assertEqual(fig.axes[0].name, '3d')
        self.assertEqual(fig.axes[0].get_ylabel(), 'time lag [days]')

    def test_invalid_kind_raises(self):
        with self.assertRaises(ValueError):
            self.V.plot(kind='pie')

    def test_wrong_axes_raises(self):
        fig, ax = plt.subplots(1, 1)

        with self.assertRaises(ValueError):
            self.V.plot(kind='scatter', ax=ax)

    def test_plotly_backend(self):
        plotting.backend('plotly')
        try:
            for kind in ('scatter', 'surface', 'contour', 'contourf', 'matrix'):
                fig = self.V.plot(kind=kind)
                self.assertGreater(len(fig.data), 0)
        finally:
            plotting.backend('matplotlib')


if __name__ == '__main__':
    unittest.main()
