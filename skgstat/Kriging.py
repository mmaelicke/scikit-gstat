"""
The kriging module offers only an Ordinary Kriging routine (OK) that can be
used together with the skgstat.Variogram class. The usage of the class is
inspired by the scipy.interpolate classes.
"""
import time

import numpy as np
from scipy.spatial.distance import squareform
from scipy.linalg import solve as scipy_solve
from numpy.linalg import solve as numpy_solve, LinAlgError, inv
from multiprocessing import Pool
import scipy.spatial.distance

from .Variogram import Variogram
from .MetricSpace import MetricSpace, MetricSpacePair

class LessPointsError(RuntimeError):
    pass


class SingularMatrixError(LinAlgError):
    pass


class IllMatrixError(RuntimeWarning):
    pass


def inv_solve(a, b):
    return inv(a).dot(b)

class OrdinaryKriging:
    def __init__(
            self,
            variogram,
            min_points=5,
            max_points=15,
            mode='exact',
            precision=100,
            solver='inv',
            n_jobs=1,
            perf=False,
            sparse=False,
            
            coordinates=None,
            values=None
    ):
        """Ordinary Kriging routine

        Ordinary kriging estimator derived from the given
        `Variogram <skgstat.Variogram>` class. To calculate estimations for
        unobserved locations, an instance of this class can either be called,
        or the `OrdinaryKriging.transform` method can be used.

        Parameters
        ----------
        variogram : Variogram
            Variogram used to build the kriging matrix. Make sure that this
            instance is describing the spatial dependence in the data well,
            otherwise the kriging estimation will most likely produce bad
            estimations.
        min_points : int
            Minimum amount of points, that have to lie within the variogram's
            range. In case not enough points are available, the estimation
            will be rejected and a null value will be estimated.
        max_points : int
            Maximum amount of points, that will be considered for the
            estimation of one unobserved location. In case more points are
            available within the variogram's range, only the `max_points`
            closest will be used for estimation. Note that the kriging matrix
            will be an max_points x max_points matrix and large numbers do
            significantly increase the calculation time.
        mode : str
            Has to be one of 'exact' or 'estimate'. In exact mode (default)
            the variogram matrix will be calculated from scratch in each
            iteration. This gives an exact solution, but it is also slower.
            In estimate mode, a set of semivariances is pre-calculated and
            the closest value will be used. This is significantly faster,
            but the estimation quality is dependent on the given precision.
        precision : int
            Only needed if `mode='estimate'`. This is the number of
            pre-calculated in-range semivariances. If chosen too low,
            the estimation will be off, if too high the performance gain is
            limited.
        solver : str
            Do not change this argument
        n_jobs : int
            Number of processes to be started in multiprocessing.
        perf : bool
            If True, the different parts of the algorithm will record their
            processing time. This is meant to be used for optimization and
            will be removed in a future version. Do not rely on this argument.
        sparse : bool

        coordinates: numpy.ndarray, MetricSpace
        values: numpy.ndarray

        """
        # store arguments to the instance

        if isinstance(variogram, Variogram):
            if coordinates is None: coordinates = variogram.coordinates
            if values is None: values = variogram.values
            variogram_descr = variogram.describe()
            if variogram_descr["model"] == "harmonize":
                variogram_descr["model"] = variogram._build_harmonized_model()
            variogram = variogram_descr
                
        self.sparse = sparse
        
        # general attributes
        self._minp = min_points
        self._maxp = max_points
        self.min_points = min_points
        self.max_points = max_points

        # general settings
        self.n_jobs = n_jobs
        self.perf = perf

        self.range = variogram['effective_range']
        self.nugget = variogram['nugget']
        self.sill = variogram['sill']
        self.dist_metric = variogram["dist_func"]
        
        # coordinates and semivariance function
        if not isinstance(coordinates, MetricSpace):
            coordinates, values = self._remove_duplicated_coordinates(coordinates, values)
            coordinates = MetricSpace(coordinates.copy(), self.dist_metric, self.range if self.sparse else None)
        else:
            assert self.dist_metric == coordinates.dist_metric, "Distance metric of variogram differs from distance metric of coordinates"
            assert coordinates.max_dist is None or coordinates.max_dist == self.range, "Sparse coordinates must have max_dist == variogram.effective_range"
        self.values = values.copy()
        self.coords = coordinates
        self.gamma_model = Variogram.fitted_model_function(**variogram)
        self.z = None

        # calculation mode; self.range has to be initialized
        self._mode = mode
        self._precision = precision
        self._prec_dist = None
        self._prec_g = None
        self.mode = mode
        self.precision = precision

        # solver settings
        self._solver = solver
        self._solve = None
        self.solver = solver

        # initialize error counter
        self.singular_error = 0
        self.no_points_error = 0
        self.ill_matrix = 0
                
        # performance counter
        if self.perf:
            self.perf_dist = list()
            self.perf_mat = list()
            self.perf_solv = list()

    def dist(self, x):
        return Variogram.wrapped_distance_function(self.dist_metric, x)
    
    @classmethod
    def _remove_duplicated_coordinates(cls, coords, values):
        """Extract the coordinates and values

        The coordinates array is checked for duplicates and only the
        first instance of a duplicate is used. Duplicated coordinates
        would result in duplicated rows in the variogram matrix and
        make it singular.

        """
        c = coords
        v = values

        _, idx = np.unique(c, axis=0, return_index=True)

        # sort the index to preserve initial order, if no duplicates were found
        idx.sort()

        return c[idx], v[idx]

    @property
    def min_points(self):
        return self._minp

    @min_points.setter
    def min_points(self, value):
        # check the value
        if not isinstance(value, int):
            raise ValueError('min_points has to be an integer.')
        if value < 0:
            raise ValueError('min_points can\'t be negative.')
        if value > self._maxp:
            raise ValueError('min_points can\'t be larger than max_points.')

        # set
        self._minp = value

    @property
    def max_points(self):
        return self._maxp

    @max_points.setter
    def max_points(self, value):
        # check the value
        if not isinstance(value, int):
            raise ValueError('max_points has to be an integer.')
        if value < 0:
            raise ValueError('max_points can\'t be negative.')
        if value < self._minp:
            raise ValueError('max_points can\'t be smaller than min_points.')

        # set
        self._maxp = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value == 'exact':
            self._prec_g = None
            self._prec_dist = None
        elif value == 'estimate':
            self._precalculate_matrix()
        else:
            raise ValueError("mode has to be one of 'exact', 'estimate'.")
        self._mode = value

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, value):
        if not isinstance(value, int):
            raise TypeError('precision has to be of type int')
        if value < 1:
            raise ValueError('The precision has be be > 1')
        self._precision = value
        self._precalculate_matrix()

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, value):
        if value == 'numpy':
            self._solve = numpy_solve
        elif value == 'scipy':
            self._solve = scipy_solve
        elif value == 'inv':
            self._solve = inv_solve
        else:
            raise AttributeError("solver has to be ['inv', 'numpy', 'scipy']")
        self._solver = value

    def transform(self, *x):
        """Kriging

        returns an estimation of the observable for the given unobserved
        locations. Each coordinate dimension should be a 1D array.

        Parameters
        ----------
        x : numpy.array, MetricSpace
            One 1D array for each coordinate dimension. Typically two or
            three array, x, y, (z) are passed for 2D and 3D Kriging

        Returns
        -------
        Z : numpy.array
            Array of estimates

        """
        # reset the internal error counters
        self.singular_error = 0
        self.no_points_error = 0
        self.ill_matrix = 0

        # reset the internal performance counter
        if self.perf:
            self.perf_dist, self.perf_mat, self.perf_solv = [], [], []

        if len(x) != 1 or not isinstance(x[0], MetricSpace):
            self.transform_coords = MetricSpace(np.column_stack(x).copy(), self.dist_metric, self.range if self.sparse else None)
        else:
            self.transform_coords = x[0]
        self.transform_coords_pair = MetricSpacePair(self.transform_coords, self.coords)
        
        # DEV: this is dirty, not sure how to do it better at the moment
        self.sigma = np.empty(len(x[0]))
        self.__sigma_index = 0
        # if multi-core, than here
        if self.n_jobs is None or self.n_jobs == 1:
            z = np.fromiter(map(self._estimator, range(len(self.transform_coords))), dtype=float)
        else:
            def f(idxs):
                return self._estimator(idxs)
            with Pool(self.n_jobs) as p:
                z = p.starmap(f, range(len(self.transform_coords)))

        # print warnings
        if self.singular_error > 0:
            print('Warning: %d kriging matrices were singular.' % self.singular_error)
        if self.no_points_error > 0:
            print('Warning: for %d locations, not enough neighbors were '
                  'found within the range.' % self.no_points_error)
        if self.ill_matrix > 0:
            print('Warning: %d kriging matrices were ill-conditioned.'
                  ' The result may not be accurate.' % self.ill_matrix)

        # store the field in the instance itself
        self.z = np.array(z)

        return np.array(z)

    def _estimator(self, idx):
        """Estimation wrapper

        Wrapper around OrdinaryKriging._krige function to build the point of
        interest for arbitrary number of dimensions. SingularMatrixError and
        LessPointsError are handled and the error counters are increased. In
        both cases numpy.NaN will be used as estimate.

        """
        try:
            z, sigma = self._krige(idx)
        except SingularMatrixError:
            self.singular_error += 1
            return np.nan
        except LessPointsError:
            self.no_points_error += 1
            return np.nan
        except IllMatrixError:
            self.ill_matrix += 1
            return np.nan

        # TODO: This is a side-effect and I need to re-design this part:
        self.sigma[self.__sigma_index] = sigma
        self.__sigma_index += 1

        return z

    def _krige(self, idx):
        """Algorithm

        Kriging algorithm for one point. This is the place, where the
        algorithm shall be changed and optimized.

        Parameters
        ----------
        idx : int
            Index into self.transform_* arrays for an unobserved location

        Raises
        ------
        SingularMatrixError:
            Raised if the kriging matrix is singular and therefore the
            equation system cannot be solved.
        LessPointsError:
            Raised if there are not the required minimum of points within the
            variogram's radius.

        Notes:
        ------

        Z is calculated as follows:

        .. math::
            \hat{Z} = \sum_i(w_i * z_i)
        
        where :math:`w_i` is the calulated kriging weight for the i-th point 
        and :math:`z_i` is the observed value at that point.

        The kriging variance :math:`\sigma^2` (sigma) is calculate as follows:

        .. math::
            \sigma^2 = \sum_i(w_i * \gamma(p_0 - p_i)) + \lambda

        where :math:`w_i` is again the weight, :math:`\gamma(p_0 - p_i)` is 
        the semivairance of the distance between the unobserved location and 
        the i-th observation. :math:`\lamda` is the Lagrange multiplier needed
        to minimize the estimation error.

        Returns
        -------
        Z : float
            estimated value at p
        sigma : float
            kriging variance :math:`\sigma^2` for p.

        """
        
        if self.perf:
            t0 = time.time()

        p = self.transform_coords.coords[idx,:]

        idx = self.transform_coords_pair.find_closest(idx, self.range, self._maxp)
        
        # raise an error if not enough points are found
        if idx.size < self._minp:
            raise LessPointsError
            
        # finally find the points and values
        in_range = self.coords.coords[idx]
        values = self.values[idx]
        dist_mat = self.coords.diagonal(idx)
        
        # if performance is tracked, time this step
        if self.perf:
            t1 = time.time()
            self.perf_dist.append(t1 - t0)

        # OLD ALGORITHM
        # a = np.ones((len(in_range) + 1, len(in_range) + 1))

        # fill; TODO: this can be done faster
        #for i in range(len(in_range)):
        #    for j in range(len(in_range)):
        #        a[i, j] = self.V.compiled_model(dist_mat[i ,j])
        # the outermost elements are all 1, except the last one
        #a[-1, -1] = 0

        # build the kriging Matrix; needs N + 1 dimensionality
        if self.mode == 'exact':
            a = self._build_matrix(dist_mat)
        else:
            a = self._estimate_matrix(dist_mat)

            # add row a column of 1's
        n = len(in_range)
        a = np.concatenate((squareform(a), np.ones((n, 1))), axis=1)
        a = np.concatenate((a, np.ones((1, n + 1))), axis=0)

        # add lagrange multiplier
        a[-1, -1] = 0

        if self.perf:
            t2 = time.time()
            self.perf_mat.append(t2 - t1)

        # build the matrix of solutions A
        _p = np.concatenate(([p], in_range))
        _dists = self.dist(_p)[:len(_p) - 1]
        _g = self.gamma_model(_dists)
        b = np.concatenate((_g, [1]))

        # solve the system
        try:
            l = self._solve(a, b)
        except LinAlgError as e:
            print(a)
            if str(e) == 'Matrix is singular.':
                raise SingularMatrixError
            else:
                raise e
        except RuntimeWarning as w:
            if 'Ill-conditioned matrix' in str(w):
                print(a)
                raise IllMatrixError
            else:
                raise w
        except ValueError as e:
            print('[DEBUG]: print variogram matrix and distance matrix:')
            print(a)
            print(_dists)
            raise e
        finally:
            if self.perf:
                t3 = time.time()
                self.perf_solv.append(t3 - t2)

        # calculate Kriging variance
        # sigma is the weights times the semi-variance to p0 
        # plus the lagrange factor 
        sigma = sum(b[:-1] * l[:-1]) + l[-1]

        # calculate Z
        Z = l[:-1].dot(values)

        # return
        return Z, sigma

    def _build_matrix(self, distance_matrix):
        # calculate the upper matrix
        return self.gamma_model(distance_matrix)

    def _precalculate_matrix(self):
        # pre-calculated distance
        self._prec_dist = np.linspace(0, self.range, self.precision)

        # pre-calculate semivariance
        self._prec_g = self.gamma_model(self._prec_dist)

    def _estimate_matrix(self, distance_matrix):
        # transform to the 'precision-space', which matches with the index
        dist_n = ((distance_matrix / self.range) * self.precision).astype(int)

        # create the gamma array
        g = np.ones(dist_n.shape) * -1

        # find all indices inside and outside the range
        out_ = np.where(dist_n >= self.precision)[0]
        in_ = np.where(dist_n < self.precision)[0]

        # all semivariances outside are set to sill,
        # the inside are estimated from the precompiled
        g[out_] = self.sill
        g[in_] = self._prec_g[dist_n[in_]]

        return g
