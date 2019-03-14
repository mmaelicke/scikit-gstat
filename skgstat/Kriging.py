"""
The kriging module offers only an Ordinary Kriging routine (OK) that can be
used together with the skgstat.Variogram class. The usage of the class is
inspired by the scipy.interpolate classes.
"""
import numpy as np
from scipy.spatial.distance import squareform
from scipy.linalg import solve, LinAlgError
from multiprocessing import Pool

from .Variogram import Variogram


class LessPointsError(RuntimeError):
    pass


class SingularMatrixError(LinAlgError):
    pass


class OrdinaryKriging:
    def __init__(
            self,
            variogram,
            min_points=5,
            max_points=15,
            solver='gen',
            n_jobs=1,
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
        solver: str
            Do not change this argument
        n_jobs = int
            Number of processes to be started in multiprocessing.

        """
        # store arguments to the instance
        if not isinstance(variogram, Variogram):
            raise TypeError('variogram has to be of type skgstat.Variogram.')

        self.V = variogram
        self._minp = min_points
        self._maxp = max_points
        self.solver = solver
        self.n_jobs = n_jobs

        # copy the distance function from the Variogram
        self.dist = self.V.dist_function
        self.range = self.V.cof[0]

        # initialize error counter
        self.singular_error = 0
        self.no_points_error = 0

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
        if value > self.max_points:
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
        if value < self.min_points:
            raise ValueError('max_points can\'t be smaller than min_points.')

        # set
        self._maxp = value

    def transform(self, *x):
        """Kriging

        returns an estimation of the observable for the given unobserved
        locations. Each coordinate dimension should be a 1D array.

        Parameters
        ----------
        x : numpy.array
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

        # if multi-core, than here.
        if self.n_jobs is None or self.n_jobs == 1:
            z = np.fromiter(map(self._estimator, *x), dtype=float)
        else:
            raise NotImplementedError
            with Pool(self.n_jobs) as p:
                z = p.starmap(self._estimator, zip(*x))

        # print warnings
        if self.singular_error > 0:
            print('Warning: %d kriging matrices were singular.' % self.singular_error)
        if self.no_points_error > 0:
            print('Warning: for %d locations, not enough neighbors were '
                  'found within the range.' % self.no_points_error)

        return np.array(z)

    def _estimator(self, *coords):
        """Estimation wrapper

        Wrapper around OrdinaryKriging._krige function to build the point of
        interest for arbitrary number of dimensions. SingularMatrixError and
        LessPointsError are handled and the error counters are increased. In
        both cases numpy.NaN will be used as estimate.

        """
        try:
            z = self._krige([*coords])
        except SingularMatrixError:
            self.singular_error += 1
            return np.nan
        except LessPointsError:
            self.no_points_error += 1
            return np.nan

        return z


    def _krige(self, p):
        """Algorithm

        Kriging algorithm for one point. This is the place, where the
        algorithm shall be changed and optimized.

        Parameters
        ----------
        p : numpy.array
            point location coordinates of the unobserved location

        Raises
        ------
        SingularMatrixError:
            Raised if the kriging matrix is singular and therefore the
            equation system cannot be solved.
        LessPointsError:
            Raised if there are not the required minimum of points within the
            variogram's radius.
        Returns
        -------
        Z : float
            estimated value at p

        """
        # determine the points needed for estimation
        _p = np.concatenate(([p], self.V.coordinates))

        # distance matrix for p to all coordinates, without p itself
        dists = squareform(self.dist(_p))[0][1:]

        # find all points within the search distance
        idx = np.where(dists <= self.range)[0]
        in_range = self.V.coordinates[idx]
        dist_mat = squareform(self.dist(in_range))
        values = self.V.values[idx]

        # check min_points and max_points parameters
        if in_range.size > self._maxp:
            in_range = in_range[np.argsort(dist_mat[0])][:self._maxp:-1]
            values = values[np.argsort(dist_mat[0])][:self._maxp:-1]
            dist_mat = squareform(self.dist(in_range))

        # min
        if in_range.size < self._minp:
            raise LessPointsError

        # build the kriging Matrix; needs N + 1 dimensionality
        a = np.ones((len(in_range) + 1, len(in_range) + 1))

        # fill; TODO: this can be done faster
        for i in range(len(in_range)):
            for j in range(len(in_range)):
                a[i, j] = self.V.compiled_model(dist_mat[i ,j])
        # the outermost elements are all 1, except the last one
        a[-1, -1] = 0

        # build the matrix of solutions A
        _p = np.concatenate(([p], in_range))
        _dists = squareform(self.dist(_p))[0][1:]
        _g = np.fromiter(map(self.V.compiled_model, _dists), dtype=float)
        b = np.concatenate((_g, [1]))

        # solve the system
        try:
            w = solve(a, b, assume_a=self.solver)
        except LinAlgError as e:
            if str(e) == 'Matrix is singular.':
                raise SingularMatrixError
            else:
                raise e

        # calulate Z
        return w[:-1].dot(values)


