import unittest
from unittest.mock import patch
import importlib
import math

import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal, assert_allclose
from scipy.spatial.distance import pdist

from skgstat import DirectionalVariogram, Variogram


class TestDirectionalVariogramInstantiation(unittest.TestCase):
    def setUp(self):
        # set up default valures, whenever c,v are not important
        np.random.seed(1306)
        self.c = np.random.gamma(11, 2, (30, 2))
        np.random.seed(1306)
        self.v = np.random.normal(5, 4, 30)

    def test_standard_settings(self):
        DV = DirectionalVariogram(self.c, self.v, normalize=True)

        assert_array_almost_equal(DV.describe()["normalized_effective_range"], 436., decimal=0)
        assert_array_almost_equal(DV.describe()["normalized_sill"], 2706., decimal=0)
        assert_array_almost_equal(DV.describe()["normalized_nugget"], 0., decimal=0)

    def test_azimuth(self):
        DV = DirectionalVariogram(self.c, self.v, azimuth=-45, normalize=True)

        assert_array_almost_equal(DV.describe()["normalized_effective_range"], 23.438, decimal=3)
        assert_array_almost_equal(DV.describe()["normalized_sill"], 219.406, decimal=3)
        assert_array_almost_equal(DV.describe()["normalized_nugget"], 0., decimal=3)

    def test_invalid_azimuth(self):
        with self.assertRaises(ValueError) as e:
            DirectionalVariogram(self.c, self.v, azimuth=360)
            self.assertEqual(
                str(e),
                'The azimuth is an angle in degree and has '
                'to meet -180 <= angle <= 180'
            )

    def test_tolerance(self):
        DV = DirectionalVariogram(self.c, self.v, tolerance=15, normalize=True)

        assert_array_almost_equal(DV.describe()["normalized_effective_range"], 435.7, decimal=1)
        assert_array_almost_equal(DV.describe()["normalized_sill"], 2722.1, decimal=1)
        assert_array_almost_equal(DV.describe()["normalized_nugget"], 0., decimal=1)

    def test_invalid_tolerance(self):
        with self.assertRaises(ValueError) as e:
            DirectionalVariogram(self.c, self.v, tolerance=-1)
            self.assertEqual(
                str(e),
                'The tolerance is an angle in degree and has to '
                'meet 0 <= angle <= 360'
            )

    def test_bandwidth(self):
        DV = DirectionalVariogram(self.c, self.v, bandwidth=12, normalize=True)

        for x, y in zip([DV.describe()[name] for name in ("normalized_effective_range", "normalized_sill", "normalized_nugget")],
                        [435.733, 2715.865, 0]):
            self.assertAlmostEqual(x, y, places=3)

    def test_invalid_model(self):
        with self.assertRaises(ValueError) as e:
            DirectionalVariogram(self.c, self.v, directional_model='NotAModel')
            self.assertEqual(
                str(e),
                'NotAModel is not a valid model.'
            )

    def test_invalid_model_type(self):
        with self.assertRaises(ValueError) as e:
            DirectionalVariogram(self.c, self.v, directional_model=5)
            self.assertEqual(
                str(e),
                'The directional model has to be identified by a '
                'model name, or it has to be the search area '
                'itself'
            )

    def test_binning_change_nlags(self):
        DV = DirectionalVariogram(self.c, self.v, n_lags=5)

        self.assertEqual(DV.n_lags, 5)

        # go through the n_lags changing procedure
        DV.bin_func = 'scott'

        # with scott, there are 6 classes now
        self.assertEqual(DV.n_lags, 6)



class Mock:
    def __init__(self, c=None, v=None):
        self._X = c
        self.values = v


class TestDirectionalVariogramMethods(unittest.TestCase):
    def setUp(self):
        # set up default valures, whenever c,v are not important
        np.random.seed(11884)
        self.c = np.random.gamma(15, 3, (30, 2))
        np.random.seed(11884)
        self.v = np.random.normal(9, 2, 30)


    def test_bin_func(self):
        DV = DirectionalVariogram(self.c, self.v, n_lags=4)
        V = Variogram(self.c, self.v, n_lags=4)

        for x, y in zip (DV.bins, V.bins):
            self.assertNotEqual(x, y)

        assert_array_almost_equal(
            np.array([12.3, 24.7, 37., 49.4]), DV.bins, decimal=1
        )

    def test_directional_mask(self):
        a = np.array([[0, 0], [1, 2], [2, 1]])

        # overwrite fit method
        class Test(DirectionalVariogram):
            def fit(*args, **kwargs):
                pass

        var = Test(a, np.random.normal(0, 1, size=3))
        var._calc_direction_mask_data()

        assert_array_almost_equal(
            np.degrees(var._angles + np.pi)[:2],
            [63.4, 26.6],
            decimal=1
        )


class TestDirectionalAngles(unittest.TestCase):
    @staticmethod
    def make(coordinates):
        class NoFit(DirectionalVariogram):
            def fit(self, *args, **kwargs):
                pass

        return NoFit(coordinates, np.arange(len(coordinates)), bandwidth=1)

    def test_pair_order_and_quadrants(self):
        coordinates = np.array([[0., 0.], [2., 1.], [-2., 1.],
                                [-2., -1.], [2., -1.], [0., 3.], [3., 0.]])
        expected = []
        for i in range(len(coordinates) - 1):
            for j in range(i + 1, len(coordinates)):
                dx, dy = coordinates[i] - coordinates[j]
                expected.append(math.atan2(dy, dx) if dy != 0 else
                                (math.pi if dx < 0 else 0.))
        var = self.make(coordinates)
        var._calc_direction_mask_data()
        assert_allclose(var._angles, expected, atol=1e-15)
        assert_array_equal(var._euclidean_dist, pdist(coordinates))

    def test_preserves_subtraction_dtype_and_extra_dimensions(self):
        coordinates = np.array([[1.1, 2.2, 3.], [4.4, 1.1, 2.],
                                [0.1, 5.5, -1.], [2.2, 3.3, 5.]])
        for dtype in [np.float16, np.float32, np.float64, np.longdouble,
                      np.int16, np.int64]:
            # Non-contiguous views must retain the same condensed order.
            c = np.repeat(coordinates.astype(dtype), 2, axis=1)[:, ::2]
            distances = pdist(c)
            dx = pdist(c[:, [0]], lambda a, b: a[0] - b[0])
            dy = pdist(c[:, [1]], lambda a, b: a[0] - b[0])
            positive = np.arccos(dx / distances)
            expected = np.where(dy >= 0, positive, -positive)
            var = self.make(c)
            var._calc_direction_mask_data()
            assert_array_equal(var._angles, expected, err_msg=str(dtype))
            assert_array_equal(var._euclidean_dist, distances)

    def test_coincident_points_and_cache(self):
        c = np.array([[0., 0.], [0., 0.], [2., 1.], [-2., -1.]])
        var = self.make(c)
        with np.errstate(invalid='ignore'):
            var._calc_direction_mask_data()
        angles = var._angles
        self.assertTrue(np.isnan(angles[0]))
        var._calc_direction_mask_data()
        self.assertIs(var._angles, angles)
        with np.errstate(invalid='ignore'):
            var._calc_direction_mask_data(force=True)
        self.assertIsNot(var._angles, angles)
        assert_array_equal(var._angles, angles)

    def test_no_python_distance_callback_per_pair(self):
        module = importlib.import_module('skgstat.DirectionalVariogram')
        var = self.make(np.arange(128).reshape(64, 2))

        def compiled_distance(coordinates, metric):
            # A callable metric invokes Python once for every pair, which
            # dominated construction time in issue #187.
            self.assertFalse(callable(metric))
            return pdist(coordinates, metric)

        with patch.object(module, 'pdist', side_effect=compiled_distance):
            var._calc_direction_mask_data()


if __name__ == '__main__':
    unittest.main()
