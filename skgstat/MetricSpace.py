from scipy.spatial.distance import pdist, cdist, squareform
from scipy.spatial import cKDTree
from scipy import sparse
import numpy as np


def _sparse_dok_get(m, fill_value=np.NaN):
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
