import copy
import numpy as np
from scipy.stats.mstats import mquantiles
from .distance import nd_dist, point_dist


def binify_even_width(X, N=10, w=None, dm=None, maxlag=None, **kwargs):
    """
    Returns a distance matrix with all entries sorted into bin numbers, along with an array of bin widths.
    The matrix has the same form as the distance matrix dm in squareform. The bins will be indexed from 0 to n.
    If dm is None, then the point_dist function will be used to calculate a distance matrix.
    kwargs will be passed to point_matrix.
    For the bins, either N or w has to be given. N specifies the number of bins and w their width.
    If both are given, N bins of width w will be specified, which might result in unexpected results.

    :param X: 1D array of x, y coordinates.
    :param N: int with the number of bins
    :param w: width of the bins
    :param dm: numpy.ndarray with the distance matrix
    :param maxlag: maximum lag for the binning
    :param kwargs: will be passed to calculate the point_matrix if no dm is given
    :return: distance matrix
    """

    _X = list(X)
    # check that all elements in the index have exactly a x and y coordinate
    if any([not len(e) == 2 for e in _X]):
        raise ValueError('The passed point data does not have a x and y coordinate for each point.')

    # get the distance matrix
    if dm is None:
        _dm = nd_dist(_X, **kwargs)
    else:
        _dm = dm

    # get the max distance
    maxval = np.max(_dm)

    # check if there was a maximum lag set
    if maxlag is not None:
        if maxlag < maxval:
            maxval = maxlag

    # either N or w has to be given
    if N is None and w is None:
        raise ValueError("Either N or w have to be given")

    # If N is not given, calculate it from maxval and width
    if N is None:
        N = int(maxval / w)

    # TODO if the user gives w only, it is ignored because the default value for N is 10

    # If N is given, calculate w from
    else:
        if w is not None:
            print('Warning! w = %d is ignored because N is already given' % w)
        w = maxval / N

    # binubound = np.cumsum(np.ones(N) * w)
    binubound = np.linspace(w, N * w, N)

    # set the last bound to the actual maxval
    binubound[-1] = np.max(_dm)

    # create bin matrix as copy of dm
    bm = copy.deepcopy(_dm)

    # set all bins except the first and last one
    for i in range(1, N - 1):
        bm[ (_dm > binubound[i - 1]) & (_dm <= binubound[i])] = i

    # set the first bin
    bm[_dm < binubound[0]] = 0

    # set the last bin
    bm[_dm > binubound[-2]] = N - 1

    # iter all bin upper bounds
#    for i, ubound in enumerate(binubound):
#        bm[ (_dm > (ubound - w)) & (_dm <= ubound) ] = i

    return np.matrix(bm), np.ones(N) * w


def binify_even_bin(X, N=10, dm=None, maxlag=None, **kwargs):
    """
    Returns a distance matrix with all entries sorted into bin numbers, along with an array of bin widths.
    The matrix has the same form as the distance matrix dm in squareform. The bins will be indexed from 0 to n.
    If dm is None, then the point_dist function will be used to calculate a distance matrix. kwargs will be passed to point_matrix.
    For the bins, either N or w has to be given. N specifies the number of bins and w their width. If both are given,
    N bins of width w will be specified, which might result in unexpected results.

    :param X: 1D array of x, y coordinates.
    :param N: int with the number of bins
    :param dm: numpy.ndarray with the distance matrix
    :param maxlag: maximum lag for the binning
    :param kwargs: will be passed to calculate the point_matrix if no dm is given
    :return:
    """

    _X = list(X)

    # check that all elements in the index have exactly a x and y coordinate
    if any([not len(e) == 2 for e in _X]):
        raise ValueError('The passed point data does not have a x and y coordinate for each point.')

    # get the distance matrix
    if dm is None:
        _dm = nd_dist(_X, **kwargs)
    else:
        _dm = dm

    # create bin matrix as copy of dm
    bm = copy.deepcopy(_dm)

    # get the upper bounds by calculating the quantiles of the upper bounds
    binubound = mquantiles(np.array(_dm).flatten(), prob=[i/N for i in range(1, N+1)])

    # set all bins except the first one
    for i in range(1, N):
        bm[ (_dm > binubound[i - 1]) & (_dm <= binubound[i]) ] = i

    # set the first bin
    bm[_dm < binubound[0]] = 0

    return np.matrix(bm), np.diff([0, *binubound])


def group_to_bin(values, bm=None, X=None, azimuth_deg=None, tolerance=22.5, maxlag=None, **kwargs):
    """
    The given values array represents values at coordinates X.
    A distance matrix is calculated and afterwards organized into bins with even width.
    This bin matrix can be given as bm; if None, The coordinates have to be given.
    For each bin, the corresponding values will be appended to a list
    The index is equal to the bin number in the returned nested list.
    The array length has to exactly fill the upper or lower triangle of the bin matrix (which is a squred triangular matrix).
    In case the azimuth_deg is not None, a irregular (or directional) Variogram will be calculated.
    This means, that only the pairs where the vector direction is within azimuth_deg +/- tolerance
    will be grouped into the bin. Otherwise they will be ignored.

    :param values: np.array with the values at the coordinates X
    :param bm:
    :param X:
    :param azimuth_deg:
    :param tolerance:
    :param maxlag:
    :param kwargs:
    :return:
    """
    # check if the input was 2D (that means, for each pair, more than one observation is used)
#    if np.array(values).ndim == 2:

    if any([isinstance(_, (tuple, list, np.ndarray)) for _ in values]):
        multidim = True
    else:
        multidim = False

    # calculate bm if needed
    if bm is None and X is None:
        raise AttributeError('If no bin matrix bm is given, you have to specify an array X of x,y coordinates.')
    elif bm is None:
        bm, bw = binify_even_width(X=X, maxlag=maxlag, **kwargs)

    # check if the azimuth_deg is given
    if azimuth_deg is not None:
        if X is None:
            raise ValueError('A directional Variogram can only be calculated, '
                             'if the coordinate tuples are given in list X.')
        if azimuth_deg < 0 or azimuth_deg >= 360:
            raise ValueError('The Azimuth has to be given in degrees.')

        # now calculate the band bounds
        up = azimuth_deg + tolerance
        lo = azimuth_deg - tolerance
        while up >= 360:
            up -= 360
        while lo < 0:
            lo += 360

    # make bm integers
    _bm = bm.astype(int)

    #  check the dimensions
    if not (len(values) == bm.shape[0] and len(values) == bm.shape[1]):
        raise ValueError('The values are not of same length as the bm axes.')

    # result container
    bin_grp = list([list() for _ in range(np.max(_bm) + 1)])

    if azimuth_deg is not None:
        dir_m = direction(X=X)

    # i is the column, j is the row, both are indexing the same points
    for i in range(len(_bm)):
        for j in range(len(_bm)):
#            if i >= j:
#                continue        # use only the upper triangle, omitting the mid-diagonal
            # if the azimuth is within the band, append otherwise ignore
            if azimuth_deg is not None:
                if _in_bounds(dir_m[i, j], lo, up):
                    if multidim:
#                        bin_grp[_bm[i, j]].extend([*values[i], *values[j]])
                        bin_grp[_bm[i, j]].extend(list(np.array(list(zip(values[i], values[j]))).flatten()))
                    else:
                        bin_grp[_bm[i, j]].extend([values[i], values[j]])

            # non - directional
            else:
                # append the values at i and j to the correct bin array
                if multidim:
                    bin_grp[_bm[i, j]].extend(list(np.array(list(zip(values[i], values[j]))).flatten()))
                else:
                    bin_grp[_bm[i, j]].extend([values[i], values[j]])

    # return
    return bin_grp


def direction(X):
    """

    :param X:
    :return:
    """
    # distance
    d = lambda p1,p2: np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    # direction-angle clockwise with Y-Axis set to 0 deg.
    def d_(p1, p2):
        x = (p2[0] - p1[0], p2[1] - p1[1])
        rad = np.arctan2(*x[::-1])
        a = (-rad * 360 / (2*np.pi)) + 90
        a += 360 if a < 0 else 0
        return a

    # this is probably not the fastest way to compute this:
    m = np.matrix(np.ones((len(X), len(X))) * np.NaN)

    for i in range(len(X)):
        for j in range(len(X)):
            if i == j:
                continue
            m[i, j] = d_(X[i], X[j])

    # return
    return m



def _in_bounds(alpha, lo, up):
    """
    Returns True, if alpha is within bounds.

    :param alpha:
    :param lo:
    :param: up:
    :return: boolean if the angle is within the boounds
    """
    if lo > up:
        return alpha >= lo or alpha <= up
    else:
        return alpha >= lo and alpha <= up