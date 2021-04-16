import scipy.spatial
import scipy.spatial.distance
import scipy.sparse
import numpy as np

def sparse_dok_get(m, fill_value=np.NaN):
    mm = np.full(m.shape, fill_value)
    for (x, y), value in m.items():
        mm[x,y] = value
    return mm

class DistanceMethods(object):
    def find_closest(self, idx, max_dist = None, N = None):
        """Find the (N) closest points (in the right set) to the point with
        index idx (in the left set).
        """
        assert self.max_dist is None or max_dist is None or max_dist == self.max_dist, "max_dist specified and max_dist != self.max_dist"
        if max_dist is None: max_dist = self.max_dist
        if isinstance(self.dists, scipy.sparse.spmatrix):
            dists = self.dists.getrow(idx)
        else:
            dists = self.dists[idx,:]
        if isinstance(dists, scipy.sparse.spmatrix):
            ridx = np.array([k[1] for k in dists.todok().keys()])
        else:
            ridx = np.where(dists <= max_dist)[0]
        if ridx.size > N:
            if isinstance(dists, scipy.sparse.spmatrix):
                selected_dists = dists[0, ridx].toarray()[0,:]
            else:
                selected_dists = dists[ridx]
            sorted_ridx = np.argsort(selected_dists, kind="stable")
            ridx = ridx[sorted_ridx][:N]
        return ridx
    
class MetricSpace(DistanceMethods):
    """A MetricSpace represents a point cloud together with a distance
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
        # Do a very quick check to see throw exceptions if self.dist_metric is invalid...
        scipy.spatial.distance.pdist(self.coords[:1,:], metric=self.dist_metric)
        

    @property
    def tree(self):
        assert self.dist_metric == "euclidean", "A coordinate tree can only be constructed for an euclidean space"
        if self._tree is None:
            self._tree = scipy.spatial.cKDTree(self.coords)
        return self._tree
            
    @property
    def dists(self):
        if self._dists is None:
            if self.max_dist is not None and self.dist_metric == "euclidean":
                self._dists = self.ltree.sparse_distance_matrix(self.tree, self.max_dist, output_type="coo_matrix").tocsr()
            else:
                self._dists = scipy.spatial.distance.squareform(
                    scipy.spatial.distance.pdist(self.coords, metric=self.dist_metric))
        return self._dists

    def diagonal(self, idx = None):
        """Return a diagonal matrix (as per
        scipy.spatial.distance.squareform), optionally for a subset of
        the points
        """        
        dist_mat = self.dists
        if idx is not None:
            dist_mat = dist_mat[idx,:][:,idx]
        if isinstance(self.dists, scipy.sparse.spmatrix):
            dist_mat = sparse_dok_get(dist_mat.todok(), np.inf)
            np.fill_diagonal(dist_mat, 0) # Normally set to inf
        return scipy.spatial.distance.squareform(dist_mat)
    
    def __len__(self):
        return len(self.coords)

class MetricSpacePair(DistanceMethods):
    """A MetricSpacePair represents a set of point clouds (MetricSpaces).
    It efficiently provides the distances between each point in one
    point cloud and each point in the other point cloud (when shorter
    than the maximum distance). The two point clouds are required to
    have the same distance metric as well as maximum distance.
    """
    def __init__(self, ms1, ms2):
        assert ms1.dist_metric == ms2.dist_metric, "Both MetricSpaces need to have the same distance metric"
        assert ms1.max_dist == ms2.max_dist, "Both MetricSpaces need to have the same max_dist"
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
        if self._dists is None:
            if self.max_dist is not None and self.dist_metric == "euclidean":
                self._dists = self.ms1.tree.sparse_distance_matrix(self.ms2.tree, self.max_dist, output_type="coo_matrix").tocsr()
            else:
                self._dists = scipy.spatial.distance.cdist(self.ms1.coords, self.ms2.coords, metric=self.ms1.dist_metric)
        return self._dists

class ProbabalisticMetricSpace(MetricSpace):
    """Like MetricSpace but samples the distance pairs only returning a
       `samples` sized subset. `samples` can either be a fraction of
       the total number of pairs (float < 1), or an integer count.
    """
    
    def __init__(self, coords, dist_metric="euclidean", max_dist=None, samples=0.5):
        self.coords = coords.copy()
        self.dist_metric = dist_metric
        self.max_dist = max_dist
        self.samples = samples

        self._lidx = None
        self._ridx = None
        self._ltree = None
        self._rtree = None
        self._dists = None
        # Do a very quick check to see throw exceptions if self.dist_metric is invalid...
        scipy.spatial.distance.pdist(self.coords[:1,:], metric=self.dist_metric)

    @property
    def sample_count(self):
        if isinstance(self.samples, int):
            return self.samples
        return int(self.samples * len(self.coords))
        
    @property
    def lidx(self):
        if self._lidx is None:
            self._lidx = np.random.choice(len(self.coords), size=self.sample_count, replace=False)
        return self._lidx
    
    @property
    def ridx(self):
        if self._ridx is None:
            self._ridx = np.random.choice(len(self.coords), size=self.sample_count, replace=False)
        return self._ridx
    
    @property
    def ltree(self):
        assert self.dist_metric == "euclidean", "A coordinate tree can only be constructed for an euclidean space"
        if self._ltree is None:
            self._ltree = scipy.spatial.cKDTree(self.coords[self.lidx,:])
        return self._ltree

    @property
    def rtree(self):
        assert self.dist_metric == "euclidean", "A coordinate tree can only be constructed for an euclidean space"
        if self._rtree is None:
            self._rtree = scipy.spatial.cKDTree(self.coords[self.ridx,:])
        return self._rtree
            
    @property
    def dists(self):
        if self._dists is None:
            max_dist = self.max_dist
            if max_dist is None:
                max_dist = 1 + np.sqrt(((self.coords.max(axis=0) - self.coords.min(axis=0))**2).sum())
            dists = self.ltree.sparse_distance_matrix(self.rtree, max_dist, output_type="coo_matrix").tocsr()
            dists.resize((len(self.coords), len(self.coords)))
            dists.indices = self.lidx[dists.indices]
            dists = dists.tocsc()
            dists.indices = self.ridx[dists.indices]
            dists = dists.tocsr()
            self._dists = dists
        return self._dists
