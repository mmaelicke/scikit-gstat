"""

"""
import numpy as np
from scipy.spatial.distance import pdist

from skgstat import binning, estimators


class SpaceTimeVariogram:
    """

    """
    def __init__(self,
                 coordinates,
                 values,
                 xdist_func='euclidean',
                 tdist_func='euclidean',
                 x_lags=10,
                 t_lags='max',
                 maxlag=None,
                 xbins='even',
                 tbins='even',
                 estimator='matheron',

                 verbose=False
                 ):
        # set coordinates array
        self._X = np.asarray(coordinates)

        # combined pairwise differences
        self._diff = None

        # set verbosity, not implemented yet
        self.verbose = verbose

        # set attributes to be fulled during calculation
        self.cov = None
        self.cof = None

        # set values
        self._values = None
        self.set_values(values=values)

        # distance matrix for space and time
        self._xdist = None
        self._tdist = None

        # set distance caluclation functions
        self._xdist_func = None
        self._tdist_func = None
        self.set_xdist_func(func_name=xdist_func)
        self.set_tdist_func(func_name=tdist_func)

        # lags and max lag
        self._x_lags = None
        self.x_lags = x_lags
        self._t_lags = None
        self.t_lags = t_lags
        self._maxlag = None
        self.maxlag = maxlag

        # estimator settings
        self._estimator = None
        self.set_estimator(estimator_name=estimator)

        # initialize binning arrays
        # space
        self._xbin_func = None
        self._xgroups = None
        self._xbins = None
        self.set_bin_func(bin_func=xbins, axis='space')

        # time
        self._tbin_func = None
        self._tgroups = None
        self._tbins = None
        self.set_bin_func(bin_func=tbins, axis='time')

        # do one preprocessing run
        self.preprocessing(force=True)

    @property
    def values(self):
        return self._values

    def set_values(self, values):
        """Set new values

        The values should be an (m, n) array with m matching the size of
        coordinates first  dimension and n is the time dimension.

        Raises
        ------
        ValueError : in case n <= 1 or values are not an array of correct
            dimensionality
        AttributeError : in case values cannot be converted to a numpy.array

        """
        values = np.asarray(values)

        # check dtype
        if not isinstance(values, np.ndarray):
            raise AttributeError('values cannot be converted to a proper '
                                 '(m,n) shaped array.')
        # check shape
        m, n = values.shape
        if m != self._X.shape[0]:
            raise ValueError('The values shape do not match coordinates.')

        if n <= 1:
            raise ValueError('A SpaceTimeVariogram needs more than one '
                             'observation on the time axis.')

        # save new values
        self._values = values

        # dismiss the pairwise differences, and lags
        self._diff = None

    @values.setter
    def values(self, new_values):
        self.set_values(values=new_values)

    @property
    def xdist_func(self):
        return self._xdist_func

    @xdist_func.setter
    def xdist_func(self, func):
        self.set_xdist_func(func_name=func)

    def set_xdist_func(self, func_name):
        """Set new space distance function

        Set a new function for calculating the distance matrix in the space
        dimension. At the moment only strings are supported. Will be passed
        to scipy.spatical.distance.pdist as 'metric' attribute.

        Parameters
        ----------
        func_name : str
            The name of the function used to calculate the pairwise distances.
            Will be passed to scipy.spatial.distance.pdist as the 'metric'
            attribute.

        Raises
        ------
        ValueError : in case a non-string argument is passed.

        """
        if isinstance(func_name, str):
            self._xdist_func = lambda x: pdist(x, metric=func_name)
        else:
            raise ValueError('For now only str arguments are supported.')

        # reset the distances
        self._xdist = None

    @property
    def tdist_func(self):
        return self._tdist_func

    @tdist_func.setter
    def tdist_func(self, func):
        self.set_tdist_func(func_name=func)

    def set_tdist_func(self, func_name):
        """Set new space distance function

        Set a new function for calculating the distance matrix in the space
        dimension. At the moment only strings are supported. Will be passed
        to scipy.spatical.distance.pdist as 'metric' attribute.

        Parameters
        ----------
        func_name : str
            The name of the function used to calculate the pairwise distances.
            Will be passed to scipy.spatial.distance.pdist as the 'metric'
            attribute.

        Raises
        ------
        ValueError : in case a non-string argument is passed.

        """
        if isinstance(func_name, str):
            self._tdist_func = lambda t: pdist(t, metric=func_name)
        else:
            raise ValueError('For now only str arguments are supported.')

        # reset the distances
        self._tdist = None

    @property
    def distance(self):
        """Distance matrices

        Returns both the space and time distance matrix. This property is
        equivalent to two separate calls of
        :func:`xdistance <skgstat.SpaceTimeVariogram.xdistance>` and
        :func:`tdistance <skgstat.SpaceTimeVariogram.tdistance>`.

        Returns
        -------
        distance matrices : (numpy.array, numpy.array)
            Returns a tuple of the two distance matrices in space and time.
            Each distance matrix is a flattened upper triangle of the
            distance matrix squareform in row orientation.

        """
        return self.xdistance, self.tdistance

    @property
    def xdistance(self):
        """Distance matrix (space)

        Return the upper triangle of the squareform pairwise distance matrix.

        Returns
        -------
        xdistance : numpy.array
            1D-array of the upper triangle of a squareform representation of
            the distance matrix.

        """
        self.__calc_xdist(force=False)
        return self._xdist

    @property
    def tdistance(self):
        """Time distance

        Returns a distance matrix containing the distance of all observation
        points in time. The time 'coordiantes' are created from the values
        multidimensional array, where the second dimension is assumed to be
        time. The unit will be time steps.

        Returns
        -------
        tdistance : numpy.array
            1D-array of the upper triangle of a squareform representation of
            the distance matrix.

        """
        self.__calc_tdist(force=False)
        return self._tdist

    @property
    def x_lags(self):
        return self._x_lags

    @x_lags.setter
    def x_lags(self, lags):
        if not isinstance(lags, int):
            raise ValueError('Only integers are supported as lag counts.')
        self._x_lags = lags

    @property
    def t_lags(self):
        if isinstance(self._t_lags, str):
            if self._t_lags.lower() == 'max':
                return self.values.shape[1]
            else:
                raise ValueError("Only 'max' supported as string argument.")
        else:
            return self._t_lags

    @t_lags.setter
    def t_lags(self, lags):
        self._t_lags = lags

    @property
    def maxlag(self):
        return self._maxlag

    @maxlag.setter
    def maxlag(self, value):
        # reset fitting
        self.cov, self.cof = None, None

        # remove binning
        self._xbins = None
        self._xgroups = None

        # set the new value
        if value is None:
            self._maxlag = None
        elif isinstance(value, str):
            if value == 'median':
                self._maxlag = np.median(self.xdistance)
            elif value == 'mean':
                self._maxlag = np.mean(self.xdistance)
        elif value < 1:
            self._maxlag = value * np.max(self.xdistance)
        else:
            self._maxlag = value

    def set_bin_func(self, bin_func, axis):
        """Set binning function

        Set a new binning function to either the space or time axis. Both axes
        support the methods: ['even', 'uniform']:

        * **'even'**, create even width bins
        * **'uniform'**, create bins of uniform distribution


        Parameters
        ----------
        bin_func : str
            Sepcifies the function to be loaded. Can be either 'even' or
            'uniform'.
        axis : str
            Specifies the axis to be used for binning. Can be either 'space' or
            'time', or one of the two shortcuts 's' and 't'

        See Also
        --------
        skgstat.binning.even_width_lags
        skgstat.binning.uniform_count_lags

        """
        # switch the function
        if bin_func.lower() == 'even':
            f = binning.even_width_lags
        elif bin_func.lower == 'uniform':
            f = binning.uniform_count_lags
        else:
            raise ValueError('%s binning method is not known' % bin_func)

        # switch the axis
        if axis.lower() == 'space' or axis.lower() == 's':
            self._xbin_func = f

            # reset
            self._xgroups = None
            self._xbins = None
        elif axis.lower() == 'time' or axis.lower() == 't':
            self._tbin_func = f

            # reset
            self._tgroups = None
            self._tbins = None
        else:
            raise ValueError('%s is not a valid axis' % axis)

        # reset fitting params
        self.cof, self.cof = None, None

    @property
    def xbins(self):
        # check if cached
        if self._xbins is None:
            self._xbins = self._xbin_func(self.xdistance,
                                          self.x_lags, self.maxlag)
        return self._xbins

    @xbins.setter
    def xbins(self, bins):
        if isinstance(bins, int):
            self._xbins = None
            self._x_lags = bins
        elif isinstance(bins, (list, tuple, np.ndarray)):
            self._xbins = np.asarray(bins)
            self._x_lags = len(self._xbins)
        elif isinstance(bins, str):
            self.set_bin_func(bin_func=bins, axis='space')
        else:
            raise AttributeError('bin value cannot be parsed.')

        # reset the groups
        self._xgroups = None

    @property
    def tbins(self):
        if self._tbins is None:
            self._tbins = self._tbin_func(self.tdistance, self.t_lags, None)

        return self._tbins

    @tbins.setter
    def tbins(self, bins):
        if isinstance(bins, int):
            self._tbins = None
            self._t_lags = bins
        elif isinstance(bins, (list, tuple, np.ndarray)):
            self._tbins = np.asarray(bins)
            self._t_lags = len(self._tbins)
        elif isinstance(bins, str):
            self.set_bin_func(bin_func=bins, axis='time')
        else:
            raise AttributeError('bin value cannot be parsed.')

        # reset the groups
        self._tgroups = None

    @property
    def meshbins(self):
        return np.meshgrid(self.xbins, self.tbins)

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self.set_estimator(estimator_name=value)

    def set_estimator(self, estimator_name):
        # reset the fitting
        self.cof, self.cov = None, None

        if isinstance(estimator_name, str):
            if estimator_name.lower() == 'matheron':
                self._estimator = estimators.matheron
            elif estimator_name.lower() == 'cressie':
                self._estimator = estimators.cressie
            elif estimator_name.lower() == 'dowd':
                self._estimator = estimators.dowd
            elif estimator_name.lower() == 'genton':
                self._estimator = estimators.genton
            elif estimator_name.lower() == 'minmax':
                self._estimator = estimators.minmax
            elif estimator_name.lower() == 'percentile':
                self._estimator = estimators.percentile
            elif estimator_name.lower() == 'entropy':
                self._estimator = estimators.entropy
            else:
                raise ValueError(
                    ('Variogram estimator %s is not understood, please' +
                    'provide the function.') % estimator_name
                )
        elif callable(estimator_name):
            self._estimator = estimator_name
        else:
            raise ValueError('The estimator has to be a string or callable.')

    def lag_groups(self, axis):
        """Lag class group mask array

        Returns a mask array for the given axis (either 'space' or 'time').
        It will have as amany elements as the respective distance matrices.
        **Unlike the base Variogram class, it does not mask the array of
        pairwise differences.**. It will mask the distance matrix of the
        respective axis.

        Parameters
        ----------
        axis : str
            Can either be 'space' or 'time'. Specifies the axis the mask array
            shall be returned for.

        Returns
        -------
        masK_array : numpy.array
            mask array that identifies the lag class group index for each pair
            of points on the given axis.

        """
        if not isinstance(axis, str):
            raise AttributeError('axis has to be of type string.')

        # space axis
        if axis.lower() == 'space' or axis.lower() == 's':
            if self._xgroups is None:
                self._calc_group(axis=axis, force=False)
            return self._xgroups

        # time axis
        elif axis.lower() == 'time' or axis.lower() == 't':
            if self._tgroups is None:
                self._calc_group(axis=axis, force=False)
            return self._tgroups

        else:
            raise ValueError("axis has to be one of 'space', 'time'.")

    def lag_classes(self):
        """Iterator over all lag classes

        Returns an iterator over all lag classes by aligning all time lags
        over all space lags. This means that it will yield all time lag groups
        for a space lag of index 0 at first and then iterate the space lags.

        Returns
        -------
        iterator

        """
        # are differences already calculated
        if self._diff is None:
            self.__calc_diff(force=False)

        # get the group masking arrays
        xgrp = self.lag_groups(axis='space')
        tgrp = self.lag_groups(axis='time')

        def diff_select(x, t):
            return self._diff[np.where(xgrp == x)[0]][:, np.where(tgrp == t)[0]]

        # iterate
        for x in range(self.x_lags):
            for t in range(self.t_lags):
                yield diff_select(x,t).flatten()

    def _get_experimental(self):
        # TODO: fix this
        if self.estimator.__name__ == 'entropy':
            raise NotImplementedError

        # this might
        z = np.fromiter(
            (self.estimator(vals) for vals in self.lag_classes()),
            dtype=float
        )

        return z.copy()

    @property
    def experimental(self):
        return self._get_experimental()

    def __calc_xdist(self, force=False):
        """Calculate distance in space

        Use :func:`xdist_func <skgstat.SpaceTimeVariogram.xdist_func>` to
        calculate the pairwise space distance matrix. The calculation will be
        cached and not recalculated. The recalculation can be foreced setting
        ``force=True``.

        Parameters
        ----------
        force : bool
            If True, an eventually cached version of the distance matrix
            will be deleted.

        """
        if self._xdist is None or force:
            self._xdist = self.xdist_func(self._X)

    def __calc_tdist(self, force=False):
        if self._tdist is None or force:
            # extract the timestamps
            t = np.stack((
                np.arange(self.values.shape[1]),
                [0] * self.values.shape[1]
            ), axis=1)
            self._tdist = self.tdist_func(t)

    def __calc_diff(self, force=False):
        # check the force
        if not force and self._diff is not None:
            return

        # get size of distance matrices
        xn = self.xdistance.size
        tn = self.tdistance.size

        # get outer and inner iterators
        outer, inner = self.values.shape
        v = self.values

        # prepare TODO: here the Fortran, Rust, whatever calc
        self._diff = np.zeros((xn, tn)) * np.nan

        xidx = 0
        for xi in range(outer):
            for xj in range(outer):
                if xi < xj:
                    tidx = 0
                    for ti in range(inner):
                        for tj in range(inner):
                            if ti < tj:
                                self._diff[xidx][tidx] = v[xi, ti] - v[xj, tj]
                                tidx += 1
                    xidx += 1

    def _calc_group(self, axis, force=False):
        # switch the axis
        if axis.lower() == 'space' or axis.lower() == 's':
            grp = self._xgroups
            fmt = 'x'
        elif axis.lower() == 'time' or axis.lower() == 't':
            grp = self._tgroups
            fmt = 't'
        else:
            raise ValueError('Axis %s is not supported' % axis)

        # check the force
        if not force and grp is not None:
            return

        # copy the arrays
        bins = getattr(self, '%sbins' % fmt)
        d = getattr(self, '%sdistance' % fmt)

        # set all groups to -1
        grp = np.ones(len(d), dtype=int) * -1

        # go for the classification
        for i, bounds in enumerate(zip([0] + list(bins), bins)):
            grp[np.where((d >= bounds[0]) & (d < bounds[1]))] = i

        # save
        setattr(self, '_%sgroups' % fmt, grp)

    def preprocessing(self, force=False):
        """

        Parameters
        ----------
        force

        Returns
        -------

        """
        # recalculate distances
        self.__calc_xdist(force=force)
        self.__calc_tdist(force=force)
        self.__calc_diff(force=force)
        self._calc_group(axis='space', force=force)
        self._calc_group(axis='time', force=force)
