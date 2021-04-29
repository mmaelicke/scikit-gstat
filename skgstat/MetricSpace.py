from scipy.spatial.distance import pdist, cdist, squareform
from scipy.spatial import cKDTree
from scipy import sparse
import numpy as np


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
