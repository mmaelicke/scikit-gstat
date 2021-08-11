from typing import Tuple, List

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.spatial import cKDTree
from scipy import sparse
import numpy as np
import multiprocessing as mp


def _sparse_dok_get(m, fill_value=np.NaN):
    """Like m.toarray(), but setting empty values to `fill_value`, by
    default `np.NaN`, rather than 0.0.

    Parameters
    ----------
    m : scipy.sparse.dok_matrix
    fill_value : float
    """
    mm = np.full(m.shape, fill_value)
    for (x, y), value in m.items():
        mm[x, y] = value
    return mm


class DistanceMethods(object):
    def find_closest(self, idx, max_dist=None, N=None):
        """find neighbors
        Find the (N) closest points (in the right set) to the point with
        index idx (in the left set).

        Parameters
        ----------
        idx : int
            Index of the point that the N closest neighbors
            are searched for.
        max_dist : float
            Maximum distance at which other points are searched
        N : int
            Number of points searched.

        Returns
        -------
        ridx : numpy.ndarray
            Indices of the N closeset points to idx

        """

        if max_dist is None:
            max_dist = self.max_dist
        else:
            if self.max_dist is not None and max_dist != self.max_dist:
                raise AttributeError(
                    "max_dist specified and max_dist != self.max_dist"
                )

        if isinstance(self.dists, sparse.spmatrix):
            dists = self.dists.getrow(idx)
        else:
            dists = self.dists[idx, :]
        if isinstance(dists, sparse.spmatrix):
            ridx = np.array([k[1] for k in dists.todok().keys()])
        elif max_dist is not None:
            ridx = np.where(dists <= max_dist)[0]
        else:
            ridx = np.arange(len(dists))
        if ridx.size > N:
            if isinstance(dists, sparse.spmatrix):
                selected_dists = dists[0, ridx].toarray()[0, :]
            else:
                selected_dists = dists[ridx]
            sorted_ridx = np.argsort(selected_dists, kind="stable")
            ridx = ridx[sorted_ridx][:N]
        return ridx


class MetricSpace(DistanceMethods):
    """
    A MetricSpace represents a point cloud together with a distance
    metric and possibly a maximum distance. It efficiently provides
    the distances between each point pair (when shorter than the
    maximum distance).

    Note: If a max_dist is specified a sparse matrix representation is
    used for the distances, which saves space and calculation time for
    large datasets, especially where max_dist << the size of the point
    cloud in space. However, it slows things down for small datasets.
    """

    def __init__(self, coords, dist_metric="euclidean", max_dist=None):
        """ProbabalisticMetricSpace class

        Parameters
        ----------
        coords : numpy.ndarray
            Coordinate array of shape (Npoints, Ndim)
        dist_metric : str
            Distance metric names as used by scipy.spatial.distance.pdist
        max_dist : float
            Maximum distance between points after which the distance
            is considered infinite and not calculated.
        """
        self.coords = coords.copy()
        self.dist_metric = dist_metric
        self.max_dist = max_dist
        self._tree = None
        self._dists = None

        # Check if self.dist_metric is valid
        try:
            pdist(self.coords[:1, :], metric=self.dist_metric)
        except ValueError as e:
            raise e

    @property
    def tree(self):
        """If `self.dist_metric` is `euclidean`, a `scipy.spatial.cKDTree`
        instance of `self.coords`. Undefined otherwise."""
        # only Euclidean supported
        if self.dist_metric != "euclidean":
            raise ValueError((
                "A coordinate tree can only be constructed "
                "for an euclidean space"
            ))

        # if not cached - calculate
        if self._tree is None:
            self._tree = cKDTree(self.coords)

        # return
        return self._tree

    @property
    def dists(self):
        """A distance matrix of all point pairs. If `self.max_dist` is
        not `None` and `self.dist_metric` is set to `euclidean`, a
        `scipy.sparse.csr_matrix` sparse matrix is returned.
        """
        # calculate if not cached
        if self._dists is None:
            # check if max dist is given
            if self.max_dist is not None and self.dist_metric == "euclidean":
                self._dists = self.tree.sparse_distance_matrix(
                    self.tree,
                    self.max_dist,
                    output_type="coo_matrix"
                ).tocsr()

            # otherwise use pdist
            else:
                self._dists = squareform(
                    pdist(self.coords, metric=self.dist_metric)
                )

        # return
        return self._dists

    def diagonal(self, idx=None):
        """
        Return a diagonal matrix (as per
        :func:`squareform <scipy.spatial.distance.squareform>`),
        optionally for a subset of the points

        Parameters
        ----------
        idx : list
            list of indices that the diagonal matrix is calculated for.

        Returns
        -------
        diagonal : numpy.ndarray
            squareform matrix of the subset of coordinates

        """
        # get the dists
        dist_mat = self.dists

        # subset dists if requested
        if idx is not None:
            dist_mat = dist_mat[idx, :][:, idx]

        # handle sparse matrix
        if isinstance(self.dists, sparse.spmatrix):
            dist_mat = _sparse_dok_get(dist_mat.todok(), np.inf)
            np.fill_diagonal(dist_mat, 0)  # Normally set to inf

        return squareform(dist_mat)

    def __len__(self):
        return len(self.coords)


class MetricSpacePair(DistanceMethods):
    """
    A MetricSpacePair represents a set of point clouds (MetricSpaces).
    It efficiently provides the distances between each point in one
    point cloud and each point in the other point cloud (when shorter
    than the maximum distance). The two point clouds are required to
    have the same distance metric as well as maximum distance.
    """
    def __init__(self, ms1, ms2):
        """
        Parameters
        ----------
        ms1 : MetricSpace
        ms2 : MetricSpace

        Note: `ms1` and `ms2` need to have the same `max_dist` and
        `distance_metric`.
        """
        # check input data
        # same distance metrix
        if ms1.dist_metric != ms2.dist_metric:
            raise ValueError(
                "Both MetricSpaces need to have the same distance metric"
            )

        # same max_dist setting
        if ms1.max_dist != ms2.max_dist:
            raise ValueError(
                "Both MetricSpaces need to have the same max_dist"
            )
        self.ms1 = ms1
        self.ms2 = ms2
        self._dists = None

    @property
    def dist_metric(self):
        return self.ms1.dist_metric

    @property
    def max_dist(self):
        return self.ms1.max_dist

    @property
    def dists(self):
        """A distance matrix of all point pairs. If `self.max_dist` is
        not `None` and `self.dist_metric` is set to `euclidean`, a
        `scipy.sparse.csr_matrix` sparse matrix is returned.
        """
        # if not cached, calculate
        if self._dists is None:
            # handle euclidean with max_dist with Tree
            if self.max_dist is not None and self.dist_metric == "euclidean":
                self._dists = self.ms1.tree.sparse_distance_matrix(
                    self.ms2.tree,
                    self.max_dist,
                    output_type="coo_matrix"
                ).tocsr()

            # otherwise Tree not possible
            else:
                self._dists = cdist(
                    self.ms1.coords,
                    self.ms2.coords,
                    metric=self.ms1.dist_metric
                )

        # return
        return self._dists


class ProbabalisticMetricSpace(MetricSpace):
    """Like MetricSpace but samples the distance pairs only returning a
       `samples` sized subset. `samples` can either be a fraction of
       the total number of pairs (float < 1), or an integer count.
    """
    def __init__(
            self,
            coords,
            dist_metric="euclidean",
            max_dist=None,
            samples=0.5,
            rnd=None
        ):
        """ProbabalisticMetricSpace class

        Parameters
        ----------
        coords : numpy.ndarray
            Coordinate array of shape (Npoints, Ndim)
        dist_metric : str
            Distance metric names as used by scipy.spatial.distance.pdist
        max_dist : float
            Maximum distance between points after which the distance
            is considered infinite and not calculated.
        samples : float, int
            Number of samples (int) or fraction of coords to sample (float < 1).
        rnd : numpy.random.RandomState, int
            Random state to use for the sampling.
        """
        self.coords = coords.copy()
        self.dist_metric = dist_metric
        self.max_dist = max_dist
        self.samples = samples
        if rnd is None:
            self.rnd = np.random
        elif isinstance(rnd, np.random.RandomState):
            self.rnd = rnd
        else:
            self.rnd = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(rnd)))

        self._lidx = None
        self._ridx = None
        self._ltree = None
        self._rtree = None
        self._dists = None
        # Do a very quick check to see throw exceptions 
        # if self.dist_metric is invalid...
        pdist(self.coords[:1, :], metric=self.dist_metric)

    @property
    def sample_count(self):
        if isinstance(self.samples, int):
            return self.samples
        return int(self.samples * len(self.coords))

    @property
    def lidx(self):
        """The sampled indices into `self.coords` for the left sample."""
        if self._lidx is None:
            self._lidx = self.rnd.choice(len(self.coords), size=self.sample_count, replace=False)
        return self._lidx

    @property
    def ridx(self):
        """The sampled indices into `self.coords` for the right sample."""
        if self._ridx is None:
            self._ridx = self.rnd.choice(len(self.coords), size=self.sample_count, replace=False)
        return self._ridx

    @property
    def ltree(self):
        """If `self.dist_metric` is `euclidean`, a `scipy.spatial.cKDTree`
        instance of the left sample of `self.coords`. Undefined otherwise."""

        # only Euclidean supported
        if self.dist_metric != "euclidean":
            raise ValueError((
                "A coordinate tree can only be constructed "
                "for an euclidean space"
            ))

        if self._ltree is None:
            self._ltree = cKDTree(self.coords[self.lidx, :])
        return self._ltree

    @property
    def rtree(self):
        """If `self.dist_metric` is `euclidean`, a `scipy.spatial.cKDTree`
        instance of the right sample of `self.coords`. Undefined otherwise."""

        # only Euclidean supported
        if self.dist_metric != "euclidean":
            raise ValueError((
                "A coordinate tree can only be constructed "
                "for an euclidean space"
            ))

        if self._rtree is None:
            self._rtree = cKDTree(self.coords[self.ridx, :])
        return self._rtree

    @property
    def dists(self):
        """A distance matrix of the sampled point pairs as a
        `scipy.sparse.csr_matrix` sparse matrix. """
        if self._dists is None:
            max_dist = self.max_dist
            if max_dist is None:
                max_dist = np.finfo(float).max
            dists = self.ltree.sparse_distance_matrix(
                self.rtree,
                max_dist,
                output_type="coo_matrix"
            ).tocsr()
            dists.resize((len(self.coords), len(self.coords)))
            dists.indices = self.ridx[dists.indices]
            dists = dists.tocsc()
            dists.indices = self.lidx[dists.indices]
            dists = dists.tocsr()
            self._dists = dists
        return self._dists


# Subfunctions used in RasterEquidistantMetricSpace 
# (outside class so that they can be pickled by multiprocessing)
def _get_disk_sample(
    coords: np.ndarray,
    center: Tuple[float, float],
    center_radius: float,
    rnd_func: np.random.RandomState,
    sample_count: int
):
    """
    Subfunction for RasterEquidistantMetricSpace.
    Calculates the indexes of a subsample in a disk "center sample".
    Same parameters as in the class.
    """
    # First index: preselect samples in a disk of certain radius
    dist_center = np.sqrt((coords[:, 0] - center[0]) ** 2 + (
            coords[:, 1] - center[1]) ** 2)
    idx1 = dist_center < center_radius

    count = np.count_nonzero(idx1)
    indices1 = np.argwhere(idx1)

    # Second index: randomly select half of the valid pixels, 
    # so that the other half can be used by the equidist
    # sample for low distances
    indices2 = rnd_func.choice(count, size=min(count, sample_count), replace=False)

    if count != 1:
        return indices1[indices2].squeeze()
    else:
        return indices1[indices2][0]


def _get_successive_ring_samples(
    coords: np.ndarray,
    center: Tuple[float, float],
    equidistant_radii: List[float],
    rnd_func: np.random.RandomState, sample_count: int
):
    """
    Subfunction for RasterEquidistantMetricSpace.
    Calculates the indexes of several subsamples within disks,
    "equidistant sample". Same parameters as in the class.
    """
    # First index: preselect samples in a ring of certain inside radius and outside radius
    dist_center = np.sqrt((coords[:, 0] - center[0]) ** 2 + (coords[:, 1] - center[1]) ** 2)

    idx = np.logical_and(
        dist_center[None, :] >= np.array(equidistant_radii[:-1])[:, None],
        dist_center[None, :] < np.array(equidistant_radii[1:])[:, None]
    )

    # Loop over an iterative sampling in rings
    list_idx = []
    for i in range(len(equidistant_radii) - 1):
        idx1 = idx[i, :]

        count = np.count_nonzero(idx1)
        indices1 = np.argwhere(idx1)

        # Second index: randomly select half of the valid pixels, so that the other half can be used by the equidist
        # sample for low distances
        indices2 = rnd_func.choice(count, size=min(count, sample_count), replace=False)
        sub_idx = indices1[indices2]

        if count > 1:
            list_idx.append(sub_idx.squeeze())
        elif count == 1:
            list_idx.append(sub_idx[0])

    return np.concatenate(list_idx)


def _get_idx_dists(
    coords: np.ndarray,
    center: Tuple[float, float],
    center_radius: float,
    equidistant_radii: List[float],
    rnd_func: np.random.RandomState,
    sample_count: int,
    max_dist: float,
    i: int,
    imax: int,
    verbose: bool
):
    """
    Subfunction for RasterEquidistantMetricSpace.
    Calculates the pairwise distances between a list of pairs of "center" and "equidistant" ensembles.
    Same parameters as in the class.
    """

    if verbose:
        print('Working on subsample ' + str(i+1) + ' out of ' + str(imax))

    cidx = _get_disk_sample(
        coords=coords, center=center,
        center_radius=center_radius,
        rnd_func=rnd_func,
        sample_count=sample_count
    )

    eqidx = _get_successive_ring_samples(
        coords=coords,
        center=center,
        equidistant_radii=equidistant_radii,
        rnd_func=rnd_func,
        sample_count=sample_count
    )

    ctree = cKDTree(coords[cidx, :])
    eqtree = cKDTree(coords[eqidx, :])

    dists = ctree.sparse_distance_matrix(
        eqtree,
        max_dist,
        output_type="coo_matrix"
    )

    return dists.data, cidx[dists.row], eqidx[dists.col]


def _mp_wrapper_get_idx_dists(argdict: dict):
    """
    Multiprocessing wrapper for get_idx_dists.
    """
    return _get_idx_dists(**argdict)


class RasterEquidistantMetricSpace(MetricSpace):
    """Like ProbabilisticMetricSpace but only applies to Raster data (2D gridded data) and
    samples iteratively an `equidistant` subset within distances to a 'center' subset.
    Subsets can either be a fraction of the total number of pairs (float < 1), or an integer count.
    The 'center' subset corresponds to a disk centered on a point of the grid for which the location
    randomly varies and can be redrawn and aggregated for several runs. The corresponding 'equidistant'
    subset consists of a concatenation of subsets drawn from rings with radius gradually increasing
    until the maximum extent of the grid is reached.

    To define the subsampling, several parameters are available:
    - The raw number of samples corresponds to the samples that will be drawn in each central disk.
     Along with the ratio of samples drawn (see below), it will automatically define the radius
     of the disk and rings for subsampling.
     Note that the number of samples drawn will be repeatedly drawn for each equidistant rings
     at a given radius, resulting in a several-fold amount of total samples for the equidistant
     subset.
    - The ratio of subsample defines the density of point sampled within each subset. It
     defaults to 20%.
    - The number of runs corresponds to the number of random center points repeated during the
    subsampling. It defaults to a sampling of 1% of the grid with center subsets.

    Alternatively, one can supply:
    - The multiplicative factor to derive increasing rings radii, set as squareroot of 2 by
    default in order to conserve a similar area for each ring and verify the sampling ratio.
    Or directly:
    - The radius of the central disk subset.
    - A list of radii for the equidistant ring subsets.
    When providing those spatial parameters, all other sampling parameters will be ignored
    except for the raw number of samples to draw in each subset.
      """

    def __init__(
            self,
            coords,
            shape,
            extent,
            samples=100,
            ratio_subsample=0.2,
            runs=None,
            n_jobs=1,
            exp_increase_fac=np.sqrt(2),
            center_radius=None,
            equidistant_radii=None,
            max_dist=None,
            dist_metric="euclidean",
            rnd=None,
            verbose=False
    ):
        """RasterEquidistantMetricSpace class

        Parameters
        ----------
        coords : numpy.ndarray
            Coordinate array of shape (Npoints, 2)
        shape : tuple[int, int]
            Shape of raster (X, Y)
        extent : tuple[float, float, float, float]
            Extent of raster (Xmin, Xmax, Ymin, Ymax)
        samples : float, int
            Number of samples (int) or fraction of coords to sample (float < 1).
        ratio_subsample:
            Ratio of samples drawn within each subsample.
        runs : int
            Number of subsamplings based on a random center point
        n_jobs : int
            Number of jobs to use in multiprocessing for the subsamplings.
        exp_increase_fac : float
            Multiplicative factor of increasing radius for ring subsets
        center_radius: float
            Radius of center subset, overrides other sampling parameters.
        equidistant_radii: list
            List of radii of ring subset, overrides other sampling parameters.
        dist_metric : str
            Distance metric names as used by scipy.spatial.distance.pdist
        max_dist : float
            Maximum distance between points after which the distance
            is considered infinite and not calculated.
        verbose : bool
            Whether to print statements in the console

        rnd : numpy.random.RandomState, int
            Random state to use for the sampling.
        """

        if dist_metric != "euclidean":
            raise ValueError((
                "A RasterEquidistantMetricSpace class can only be constructed "
                "for an euclidean space"
            ))

        self.coords = coords.copy()
        self.dist_metric = dist_metric
        self.shape = shape
        self.extent = extent
        self.res = np.mean([(extent[1] - extent[0])/(shape[0]-1),(extent[3] - extent[2])/(shape[1]-1)])

        # if the maximum distance is not specified, find the maximum possible distance from the extent
        if max_dist is None:
            max_dist = np.sqrt((extent[1] - extent[0])**2 + (extent[3] - extent[2])**2)
        self.max_dist = max_dist

        self.samples = samples

        if runs is None:
            # If None is provided, try to sample center samples for about one percent of the area
            runs = int((self.shape[0] * self.shape[1]) / self.samples * 1/100.)
        self.runs = runs

        self.n_jobs = n_jobs

        if rnd is None:
            self.rnd = np.random.default_rng()
        elif isinstance(rnd, np.random.RandomState):
            self.rnd = rnd
        else:
            self.rnd = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(rnd)))

        # Radius of center subsample, based on sample count
        # If None is provided, the disk is defined with the exact size to hold the number of percentage of samples
        # defined by the user
        if center_radius is None:
            center_radius = np.sqrt(1. / ratio_subsample * self.sample_count / np.pi) * self.res
            if verbose:
                print('Radius of center disk sample for sample count of '+str(self.sample_count)+ ' and subsampling ratio'
                      ' of '+str(ratio_subsample)+': '+str(center_radius))
        self._center_radius = center_radius

        # Radii of equidistant ring subsamples
        # If None is provided, the rings are defined with exponentially increasing radii with a factor sqrt(2), which
        # means each ring will have just enough area to sample at least the number of samples desired, and same
        # for each of the following, due to:
        # (sqrt(2)R)**2 - R**2 = R**2
        if equidistant_radii is None:
            equidistant_radii = [0.]
            increasing_rad = self._center_radius
            while increasing_rad < self.max_dist:
                equidistant_radii.append(increasing_rad)
                increasing_rad *= exp_increase_fac
            equidistant_radii.append(self.max_dist)
            if verbose:
                print('Radii of equidistant ring samples for increasing factor of ' + str(exp_increase_fac) + ': ')
                print(equidistant_radii)
        self._equidistant_radii = equidistant_radii

        self.verbose = verbose

        # Index and KDTree of center sample
        self._cidx = None
        self._ctree = None

        # Index and KDTree of equidistant sample
        self._eqidx = None
        self._eqtree = None

        self._centers = None
        self._dists = None
        # Do a very quick check to see throw exceptions
        # if self.dist_metric is invalid...
        pdist(self.coords[:1, :], metric=self.dist_metric)

    @property
    def sample_count(self):
        if isinstance(self.samples, int):
            return self.samples
        return int(self.samples * len(self.coords))

    @property
    def cidx(self):
        """The sampled indices into `self.coords` for the center sample."""
        return self._cidx

    @property
    def ctree(self):
        """If `self.dist_metric` is `euclidean`, a `scipy.spatial.cKDTree`
        instance of the center sample of `self.coords`. Undefined otherwise."""

        # only Euclidean supported
        if self.dist_metric != "euclidean":
            raise ValueError((
                "A coordinate tree can only be constructed "
                "for an euclidean space"
            ))

        if self._ctree is None:
            self._ctree = [cKDTree(self.coords[self.cidx[i], :]) for i in range(len(self.cidx))]
        return self._ctree


    @property
    def eqidx(self):
        """The sampled indices into `self.coords` for the equidistant sample."""
        return self._eqidx

    @property
    def eqtree(self):
        """If `self.dist_metric` is `euclidean`, a `scipy.spatial.cKDTree`
        instance of the equidistant sample of `self.coords`. Undefined otherwise."""

        # only Euclidean supported

        if self._eqtree is None:
            self._eqtree = [cKDTree(self.coords[self.eqidx[i], :]) for i in range(len(self.eqidx))]
        return self._eqtree

    @property
    def dists(self):
        """A distance matrix of the sampled point pairs as a
        `scipy.sparse.csr_matrix` sparse matrix. """

        # Derive distances
        if self._dists is None:

            idx_center = self.rnd.choice(len(self.coords), size=min(self.runs, len(self.coords)), replace=False)

            # Each run has a different center
            centers = self.coords[idx_center]

            # Running on a single core: for loop
            if self.n_jobs == 1:

                list_dists, list_cidx, list_eqidx = ([] for i in range(3))

                for i in range(self.runs):

                    center = centers[i]
                    dists, cidx, eqidx = _get_idx_dists(self.coords, center=center, center_radius=self._center_radius,
                                                       equidistant_radii=self._equidistant_radii, rnd_func=self.rnd,
                                                       sample_count=self.sample_count, max_dist=self.max_dist, i=i,
                                                       imax=self.runs, verbose=self.verbose)
                    list_dists.append(dists)
                    list_cidx.append(cidx)
                    list_eqidx.append(eqidx)

            # Running on several cores: multiprocessing
            else:
                # Arguments to pass: only centers and loop index for verbose are changing
                argsin = [{'center': centers[i], 'coords': self.coords, 'center_radius': self._center_radius,
                           'equidistant_radii': self._equidistant_radii, 'rnd_func': self.rnd,
                           'sample_count': self.sample_count, 'max_dist': self.max_dist, 'i': i, 'imax': self.runs,
                           'verbose': self.verbose} for i in range(self.runs)]

                # Process in parallel
                pool = mp.Pool(self.n_jobs, maxtasksperchild=1)
                outputs = pool.map(_mp_wrapper_get_idx_dists, argsin, chunksize=1)
                pool.close()
                pool.join()

                # Get lists of outputs
                list_dists, list_cidx, list_eqidx = list(zip(*outputs))

            # Define class objects
            self._centers = centers
            self._cidx = list_cidx
            self._eqidx = list_eqidx

            # concatenate the coo matrixes
            d = np.concatenate(list_dists)
            c = np.concatenate(list_cidx)
            eq = np.concatenate(list_eqidx)

            # remove possible duplicates (that would be summed by default)
            # from https://stackoverflow.com/questions/28677162/ignoring-duplicate-entries-in-sparse-matrix

            # Stable solution but a bit slow
            # c, eq, d = zip(*set(zip(c, eq, d)))
            # dists = sparse.csr_matrix((d, (c, eq)), shape=(len(self.coords), len(self.coords)))

            # Solution 5+ times faster than the preceding, but relies on _update() which might change in scipy (which
            # only has an implemented method for summing duplicates, and not ignoring them yet)
            dok = sparse.dok_matrix((len(self.coords), len(self.coords)))
            dok._update(zip(zip(c, eq), d))
            dists = dok.tocsr()

            self._dists = dists

        return self._dists