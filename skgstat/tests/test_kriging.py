import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
from skgstat import Variogram, OrdinaryKriging


class TestKrigingInstantiation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.c = np.random.gamma(10, 4, size=(50, 2))
        np.random.seed(42)
        self.v = np.random.normal(10, 2, size=50)
        self.V = Variogram(self.c, self.v, model='gaussian', normalize=False)

    def test_coordinates_and_values(self):
        ok = OrdinaryKriging(self.V)
        assert_array_almost_equal(self.c, ok.coords.coords)

    def test_coordinates_with_duplicates(self):
        c = self.c.copy()

        # create two duplicates
        c[14] = c[42]
        c[8] = c[42]

        V = Variogram(c, self.v)
        ok = OrdinaryKriging(V)

        # two instances should be removed
        self.assertEqual(len(ok.coords), 50 - 2)

    def test_min_points_type_check(self):
        with self.assertRaises(ValueError) as e:
            OrdinaryKriging(self.V, min_points=4.0)
        
        self.assertEqual(
            str(e.exception), 'min_points has to be an integer.'
        )

    def test_min_points_negative(self):
        with self.assertRaises(ValueError) as e:
            OrdinaryKriging(self.V, min_points=-2)
        
        self.assertEqual(
            str(e.exception), 'min_points can\'t be negative.'
        )

    def test_min_points_larger_max_points(self):
        with self.assertRaises(ValueError) as e:
            OrdinaryKriging(self.V, min_points=10, max_points=5)
        
        self.assertEqual(
            str(e.exception), 'min_points can\'t be larger than max_points.'
        )

    def test_max_points_type_check(self):
        with self.assertRaises(ValueError) as e:
            OrdinaryKriging(self.V, max_points=16.0)
        
        self.assertEqual(
            str(e.exception), 'max_points has to be an integer.'
        )

    def test_max_points_negative(self):
        with self.assertRaises(ValueError) as e:
            ok = OrdinaryKriging(self.V, max_points=10)
            ok.max_points = - 2
            
        self.assertEqual(
            str(e.exception), 'max_points can\'t be negative.'
        )

    def test_max_points_smaller_min_points(self):
        with self.assertRaises(ValueError) as e:
            ok = OrdinaryKriging(self.V, min_points=3, max_points=5)
            ok.max_points = 2
        
        self.assertEqual(
            str(e.exception), 'max_points can\'t be smaller than min_points.'
        )

    def test_mode_settings(self):
        # estimate mode
        ok = OrdinaryKriging(self.V, mode='estimate')
        self.assertIsNotNone(ok._prec_g)
        self.assertIsNotNone(ok._prec_dist)

        # exact mode
        ok.mode = 'exact'
        self.assertIsNone(ok._prec_g)
        self.assertIsNone(ok._prec_dist)

    def test_mode_unknown(self):
        with self.assertRaises(ValueError) as e:
            OrdinaryKriging(self.V, mode='foo')
            
        self.assertEqual(
            str(e.exception), "mode has to be one of 'exact', 'estimate'."
        )

    def test_precision_TypeError(self):
        with self.assertRaises(TypeError) as e:
            OrdinaryKriging(self.V, precision='5.5')
            
        self.assertEqual(
            str(e.exception), 'precision has to be of type int'
        )

    def test_precision_ValueError(self):
        with self.assertRaises(ValueError) as e:
            OrdinaryKriging(self.V, precision=0)
            
        self.assertEqual(
            str(e.exception), 'The precision has be be > 1'
        )

    def test_solver_AttributeError(self):
        with self.assertRaises(AttributeError) as e:
            OrdinaryKriging(self.V, solver='peter')
            
        self.assertEqual(
            str(e.exception), "solver has to be ['inv', 'numpy', 'scipy']"
        )


class TestPerformance(unittest.TestCase):
    """
    The TestPerformance class is not a real unittest. It will always be true.
    It does apply some benchmarking, which could be included into the testing
    framework, as soon as the OrdinaryKriging class is finalized. From that
    point on, new code should not harm the performance significantly.
    """
    def setUp(self):
        # define the target field
        def func(x, y):
            return np.sin(0.02 * np.pi * y) * np.cos(0.02 * np.pi * x)

        # create a grid
        self.grid_x, self.grid_y = np.mgrid[0:100:100j, 0:100:100j]

        # sample the field
        np.random.seed(42)
        self.x = np.random.randint(100, size=300)
        np.random.seed(1337)
        self.y = np.random.randint(100, size=300)
        self.z = func(self.x, self.y)

        # build the Variogram and Kriging class
        self.V = Variogram(list(zip(self.x, self.y)), self.z,
                           model='exponential',
                           n_lags=15,
                           maxlag=0.4,
                           normalize=False
                           )
        self.ok = OrdinaryKriging(self.V, min_points=2, max_points=5, perf=True)

    def _run_benchmark(self, points):
        xi = self.grid_x.flatten()[:points]
        yi = self.grid_y.flatten()[:points]

        # run
        res = self.ok.transform(xi, yi)
        self.ok.perf_dist *= 1000
        self.ok.perf_mat *= 1000
        self.ok.perf_solv *= 1000

        print('Benchmarking OrdinaryKriging...')
        print('-------------------------------')
        print('Points:', points)
        print('Solver:', self.ok.solver)
        print('Mode:', self.ok.mode)
        print('Build distance matrix:  %.1f ms (%.4f ms each)' %
              (np.sum(self.ok.perf_dist), np.std(self.ok.perf_dist)))
        print('Build variogram matrix: %.1f ms (%.4f ms each)' %
              (np.sum(self.ok.perf_mat), np.std(self.ok.perf_mat)))
        print('Solve kriging matrix:   %.1f ms (%.4f ms each)' %
              (np.sum(self.ok.perf_solv), np.std(self.ok.perf_solv)))
        print('---------------------------------------------')

    def test_200points_exact(self):
        self.ok.mode = 'exact'
        self.ok.solver = 'inv'
        self._run_benchmark(points=200)

    def test_2000points_exact(self):
        self.ok.mode = 'exact'
        self.ok.solver = 'inv'
        self._run_benchmark(points=2000)

    def test_200points_estimate(self):
        self.ok.mode = 'estimate'
        self.ok.solver = 'inv'
        self._run_benchmark(points=200)

    def test_2000points_estimate(self):
        self.ok.mode = 'estimate'
        self.ok.solver = 'inv'
        self._run_benchmark(points=2000)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
