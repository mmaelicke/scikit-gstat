"""
Spatio-temporal variogram for observations located freely in space and time.

"""
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.optimize import curve_fit

from skgstat import binning, estimators, models, stmodels, plotting


class SpatioTemporalVariogram:
    """Spatio-temporal Variogram

    .. versionadded:: 1.1

    Estimates an experimental space-time variogram from observations that
    are located freely in space and time. Each observation is one row of
    the coordinates array: the last column holds the time, all preceding
    columns are spatial coordinates. In contrast to
    :class:`SpaceTimeVariogram <skgstat.SpaceTimeVariogram>`, the
    observations do not need to form a complete location-by-time grid.
    A complete grid can be converted using
    :func:`from_grid <skgstat.SpatioTemporalVariogram.from_grid>`.

    All observation pairs are used. Thus, pairs at the same location and
    pairs observed at the same time contribute to the experimental
    variogram and form the two marginal variograms.

    Parameters
    ----------
    coordinates : numpy.ndarray, pandas.DataFrame
        Array of shape (n, d + 1). The first d columns are the spatial
        coordinates, the last column is time. Time can be numeric or any
        datetime-like type (datetime, numpy.datetime64, pandas.Timestamp).
    values : numpy.ndarray
        Array of shape (n, ) holding one observation per coordinate row.
    xdist_func : str
        Any metric supported by :func:`scipy.spatial.distance.pdist`. The
        metric is only applied to the spatial coordinates.
    x_lags : int
        Number of space lag classes.
    t_lags : int
        Number of time lag classes.
    maxlag_space : float, str
        Maximum space lag. Follows the rules of
        :func:`Variogram.maxlag <skgstat.Variogram.maxlag>`: a number
        ``0 < maxlag < 1`` is relative to the maximum separating distance,
        larger numbers are absolute. ``'mean'`` and ``'median'`` use the
        mean or median distance. If None, all pairs are used.
    maxlag_time : float, str
        Maximum time lag. Same rules as for ``maxlag_space``. Absolute
        values are given in the internal time unit, which is seconds for
        datetime-like input.
    xbins : str, Iterable, Callable
        Binning of the space axis. Can be ``'even'``, ``'uniform'``,
        ``'kmeans'``, ``'ward'`` or any method supported by
        :func:`numpy.histogram_bin_edges`. Alternatively, a function of
        signature ``func(distances, n, maxlag)`` or the upper bin edges
        can be passed.
    tbins : str, Iterable, Callable
        Binning of the time axis. Same options as for ``xbins``.
    estimator : str, Callable
        Semi-variance estimator. Can be one of ``'matheron'``,
        ``'cressie'``, ``'dowd'``, ``'genton'``, ``'minmax'``,
        ``'percentile'``, ``'entropy'`` or a function.
    use_nugget : bool
        If True, the marginal models are fitted with a nugget.
    model : str
        Space-time model. Has to be one of ``'sum'``, ``'product'`` or
        ``'product-sum'``.
    marginal_model : str
        Theoretical model fitted to both marginal variograms. Has to be one
        of ``'spherical'``, ``'exponential'``, ``'gaussian'``,
        ``'cubic'``, ``'stable'`` or ``'matern'``.

    Notes
    -----
    Datetime-like time is anchored at the oldest timestamp and stored as
    elapsed seconds. All lags, bins and fitted models use seconds; only the
    plots convert the time axis into the unit implied by the data
    resolution. Numeric time is used as it is.

    """
    def __init__(self,
                 coordinates,
                 values,
                 xdist_func='euclidean',
                 x_lags=10,
                 t_lags=10,
                 maxlag_space=None,
                 maxlag_time=None,
                 xbins='even',
                 tbins='even',
                 estimator='matheron',
                 use_nugget=False,
                 model='product-sum',
                 marginal_model='spherical'
                 ):
        # check the number of lags
        if not isinstance(x_lags, int) or x_lags < 1:
            raise ValueError('x_lags has to be a positive integer.')
        if not isinstance(t_lags, int) or t_lags < 1:
            raise ValueError('t_lags has to be a positive integer.')
        if not isinstance(use_nugget, bool):
            raise ValueError('use_nugget has to be a boolean value.')

        # set coordinates, time and values
        self._X = None
        self._time = None
        self._time_origin = None
        self._time_unit = None
        self._values = None
        self._set_observations(coordinates, values)

        # set estimator and models
        self._estimator = self._get_estimator(estimator)
        self._model_name, self._model = self._get_model(model)
        self._marginal_model_name, self._marginal_model = \
            self._get_marginal_model(marginal_model)
        self._use_nugget = use_nugget

        # set attributes to be filled during fitting
        self.cof = None
        self.cov = None
        self._marginal_cof = {}
        self._marginal_cov = {}
        self._model_params = {}

        # pairwise distances in space and time, and value differences
        self._xdist = pdist(self._X, metric=xdist_func)
        self._tdist = pdist(self._time.reshape(-1, 1), metric='cityblock')
        self._diff = pdist(self._values.reshape(-1, 1), metric='cityblock')

        # maximum lags
        self._maxlag_space = self._get_maxlag(maxlag_space, self._xdist)
        self._maxlag_time = self._get_maxlag(maxlag_time, self._tdist)

        # bin edges
        self._xbins = self._get_bins(
            xbins, self._xdist, x_lags, self._maxlag_space, 'space'
        )
        self._tbins = self._get_bins(
            tbins, self._tdist, t_lags, self._maxlag_time, 'time'
        )

        # lag class groups
        self._xgroups = np.digitize(self._xdist, self._xbins, right=True)
        self._tgroups = np.digitize(self._tdist, self._tbins, right=True)

        # experimental variogram
        self._experimental = None
        self._bin_count = None
        self._calc_experimental()

    @classmethod
    def from_grid(cls, coordinates, values, times=None, **kwargs):
        """Create from a location-by-time grid

        Convert a complete location-by-time grid, as used by
        :class:`SpaceTimeVariogram <skgstat.SpaceTimeVariogram>`, into one
        observation per grid cell. Non-finite grid cells are treated as
        missing observations and dropped.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Array of shape (m, d) holding the spatial coordinates of the
            m locations.
        values : numpy.ndarray
            Array of shape (m, n) holding a time series of length n for
            each location.
        times : Iterable
            The n observation times. Can be numeric or datetime-like. If
            None, ``0, 1, ..., n - 1`` will be used.
        kwargs : dict
            All other keyword arguments are passed to
            :class:`SpatioTemporalVariogram <skgstat.SpatioTemporalVariogram>`.

        Returns
        -------
        variogram : SpatioTemporalVariogram

        Notes
        -----
        The resulting experimental variogram is not equal to the one of
        :class:`SpaceTimeVariogram <skgstat.SpaceTimeVariogram>`, as the
        latter does not use pairs at the same location, pairs at the same
        time and only one direction of each location-time pair.

        """
        coordinates = np.asarray(coordinates, dtype=float)
        values = np.asarray(values, dtype=float)

        # check the grid
        if coordinates.ndim != 2:
            raise ValueError('coordinates has to be a two dimensional array.')
        if values.ndim != 2 or values.shape[0] != len(coordinates):
            raise ValueError('values has to be of shape (n_locations, n_times).')

        # create the time axis
        if times is None:
            times = np.arange(values.shape[1], dtype=float)
        times = np.asarray(times)
        if times.ndim != 1 or len(times) != values.shape[1]:
            raise ValueError('times has to match the second axis of values.')

        # one row per grid cell: locations change slowest, time fastest
        # use a DataFrame, as numpy turns datetime64[ns] into int in object arrays
        n_loc, n_time = values.shape
        X = pd.DataFrame(np.repeat(coordinates, n_time, axis=0))
        X['time'] = np.tile(times, n_loc)

        # drop missing observations
        values = values.flatten()
        idx = np.isfinite(values)

        return cls(X.loc[idx], values[idx], **kwargs)

    # ----------------------------------------------------------------------- #
    #                        ATTRIBUTE SETTING                                #
    # ----------------------------------------------------------------------- #
    def _set_observations(self, coordinates, values):
        """Set coordinates, time and values

        Splits the coordinates into space and time, converts datetime-like
        time into elapsed seconds and validates the observations.

        Raises
        ------
        ValueError : in case the shapes do not match, the input contains
            non-finite values or duplicated space-time coordinates.

        """
        coordinates = np.asarray(coordinates)
        values = np.asarray(values, dtype=float)

        # check shapes
        if coordinates.ndim != 2 or coordinates.shape[1] < 2:
            raise ValueError(
                'coordinates has to be of shape (n, d + 1) with at least one '
                'spatial column and the time as last column.'
            )
        if len(coordinates) < 2:
            raise ValueError('At least two observations are needed.')
        if values.ndim != 1 or len(values) != len(coordinates):
            raise ValueError(
                'values has to be one dimensional and match the coordinates.'
            )

        # split space and time
        X = np.asarray(coordinates[:, :-1], dtype=float)
        time, origin = self._normalize_time(coordinates[:, -1])

        # check for finite input
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(time)) or \
                not np.all(np.isfinite(values)):
            raise ValueError('coordinates and values have to be finite.')

        # check for duplicated space-time coordinates
        _, inverse, counts = np.unique(
            np.column_stack((X, time)),
            axis=0,
            return_inverse=True,
            return_counts=True
        )
        if np.any(counts > 1):
            rows = np.flatnonzero(inverse.flatten() == np.argmax(counts > 1))
            raise ValueError(
                'Duplicated space-time coordinates found at rows %s.' %
                rows.tolist()
            )

        self._X = X
        self._time = time
        self._time_origin = origin
        self._time_unit = self._infer_time_unit(time) if origin is not None else None
        self._values = values

    @staticmethod
    def _normalize_time(time):
        """Convert the time axis

        Datetime-like time is converted into seconds elapsed since the
        oldest timestamp, which is returned as origin. Numeric time is
        returned as it is, with origin None.

        """
        time = np.asarray(time)

        # check if the time axis is datetime-like
        is_datetime = np.issubdtype(time.dtype, np.datetime64) or (
            time.dtype == object and
            all(isinstance(t, (date, datetime, np.datetime64, pd.Timestamp)) for t in time)
        )

        if not is_datetime:
            return np.asarray(time, dtype=float), None

        # anchor at the oldest timestamp
        timestamps = pd.DatetimeIndex(pd.to_datetime(time))
        if timestamps.hasnans:
            raise ValueError('The time axis must not contain NaT.')
        origin = timestamps.min()
        seconds = (timestamps - origin).total_seconds().to_numpy(dtype=float)

        return seconds, origin

    @staticmethod
    def _infer_time_unit(time):
        """Infer the plotting unit from the time resolution in seconds"""
        resolution = np.diff(np.unique(time))
        if resolution.size == 0:
            return 'seconds'

        # use the largest unit the resolution is a multiple of
        resolution = np.min(resolution)
        for unit, seconds in (('days', 86400.), ('hours', 3600.), ('minutes', 60.)):
            if resolution >= seconds and np.isclose(resolution % seconds, 0.):
                return unit
        return 'seconds'

    @staticmethod
    def _get_estimator(estimator):
        if callable(estimator):
            return estimator

        names = ('matheron', 'cressie', 'dowd', 'genton', 'minmax', 'percentile', 'entropy')
        if not isinstance(estimator, str) or estimator.lower() not in names:
            raise ValueError(
                'Variogram estimator %s is not understood, please provide '
                'the function.' % estimator
            )
        return getattr(estimators, estimator.lower())

    @staticmethod
    def _get_model(model):
        if isinstance(model, str):
            name = model.lower().replace('_', '-')
            if name == 'sum':
                return name, stmodels.sum
            elif name == 'product':
                return name, stmodels.product
            elif name == 'product-sum':
                return name, stmodels.product_sum

        raise ValueError(
            "%s is not a valid space-time model. Use one of ['sum', "
            "'product', 'product-sum']." % model
        )

    @staticmethod
    def _get_marginal_model(model):
        names = ('spherical', 'exponential', 'gaussian', 'cubic', 'stable', 'matern')
        if not isinstance(model, str) or model.lower() not in names:
            raise ValueError(
                'The theoretical Variogram function %s is not understood. '
                'Use one of %s.' % (model, list(names))
            )
        return model.lower(), getattr(models, model.lower())

    @staticmethod
    def _get_maxlag(maxlag, distances):
        """Resolve maxlag like :func:`Variogram.maxlag <skgstat.Variogram.maxlag>`"""
        if maxlag is None:
            return np.max(distances)
        elif isinstance(maxlag, str):
            if maxlag == 'median':
                return np.median(distances)
            elif maxlag == 'mean':
                return np.mean(distances)
            raise ValueError("maxlag has to be a number, 'mean' or 'median'.")
        elif maxlag <= 0:
            raise ValueError('maxlag has to be larger than zero.')
        elif maxlag < 1:
            return maxlag * np.max(distances)
        return float(maxlag)

    @staticmethod
    def _get_bins(bin_func, distances, n, maxlag, axis):
        """Calculate the upper bin edges for one axis"""
        # an axis without separation collapses into one zero lag class
        if np.all(distances == 0):
            return np.array([0.])

        if isinstance(bin_func, str):
            fname = bin_func.lower()
            if fname == 'even':
                bins, _ = binning.even_width_lags(distances, n, maxlag)
            elif fname == 'uniform':
                bins, _ = binning.uniform_count_lags(distances, n, maxlag)
            elif fname == 'kmeans':
                bins, _ = binning.kmeans(distances, n, maxlag)
            elif fname == 'ward':
                bins, _ = binning.ward(distances, n, maxlag)
            else:
                bins, _ = binning.auto_derived_lags(distances, fname, maxlag)
        elif callable(bin_func):
            bins, _ = bin_func(distances, n, maxlag)
        else:
            bins = bin_func

        # check the bin edges
        bins = np.asarray(bins, dtype=float)
        if bins.ndim != 1 or bins.size == 0 or not np.all(np.isfinite(bins)) \
                or np.any(bins < 0) or np.any(np.diff(bins) <= 0):
            raise ValueError(
                'The %s bin edges have to be a finite, increasing, one '
                'dimensional array.' % axis
            )
        return bins

    # ----------------------------------------------------------------------- #
    #                        PRE-PROCESSING                                   #
    # ----------------------------------------------------------------------- #
    def _calc_experimental(self):
        """Calculate the experimental variogram

        The lag classes are left-open intervals ``(lower, upper]``, like in
        :class:`SpaceTimeVariogram <skgstat.SpaceTimeVariogram>`, but the
        zero lags are included in the first class. Pairs beyond the last
        bin edge are not used.

        """
        self._experimental = np.ones((self.x_lags, self.t_lags)) * np.nan
        self._bin_count = np.zeros((self.x_lags, self.t_lags), dtype=int)

        for i in range(self.x_lags):
            for j in range(self.t_lags):
                idx = (self._xgroups == i) & (self._tgroups == j)
                self._bin_count[i, j] = np.count_nonzero(idx)
                self._experimental[i, j] = self._estimator(self._diff[idx])

    def get_marginal(self, axis):
        """Marginal experimental variogram

        Calculates the experimental marginal variogram along the given
        axis. The space marginal only uses pairs observed at the same time,
        the time marginal only uses pairs observed at the same location.

        Parameters
        ----------
        axis : str
            Either ``'space'`` or ``'time'``.

        Returns
        -------
        marginal : numpy.ndarray
            Semi-variances for each lag class of the given axis.

        Raises
        ------
        ValueError : in case there are no pairs at zero lag on the other
            axis, which is common for irregularly sampled data.

        """
        if axis.lower() in ('space', 's'):
            idx = self._tdist == 0
            groups, n = self._xgroups, self.x_lags
        elif axis.lower() in ('time', 't'):
            idx = self._xdist == 0
            groups, n = self._tgroups, self.t_lags
        else:
            raise ValueError("axis can either be 'space' or 'time'.")

        if not np.any(idx):
            raise ValueError(
                'The %s marginal needs pairs at zero lag on the other axis, '
                'but there are none.' % axis
            )

        return np.fromiter(
            (self._estimator(self._diff[idx & (groups == i)]) for i in range(n)),
            dtype=float
        )

    # ----------------------------------------------------------------------- #
    #                               FITTING                                   #
    # ----------------------------------------------------------------------- #
    def _fit_marginal(self, axis):
        """Fit the marginal model to the experimental marginal variogram"""
        y = self.get_marginal(axis)
        x = self._xbins if axis == 'space' else self._tbins

        # use only filled lag classes off the origin
        idx = np.isfinite(y) & (x > 0)
        x, y = x[idx], y[idx]

        # check that there are enough lag classes for all parameters
        n_params = 2 + int(self._marginal_model_name in ('stable', 'matern')) + int(self._use_nugget)
        if x.size < n_params:
            raise ValueError(
                'The %s marginal has not enough filled lag classes to be '
                'fitted.' % axis
            )

        # upper bounds as in Variogram: range, sill, shape and nugget
        bounds = [np.max(x), np.max(y)]
        if self._marginal_model_name == 'stable':
            bounds.append(2.)
        elif self._marginal_model_name == 'matern':
            bounds.append(20.)
        if self._use_nugget:
            bounds.append(0.99 * np.max(y))

        # wrap the model to include or exclude the nugget
        if self._use_nugget:
            def wrapped(*args):
                return self._marginal_model(*args)
        else:
            def wrapped(*args):
                return self._marginal_model(*args, 0)

        return curve_fit(
            wrapped,
            x, y,
            method='trf',
            p0=np.asarray(bounds),
            bounds=(0, bounds)
        )

    def _marginal(self, axis, h):
        """Apply the fitted marginal model"""
        cof = self._marginal_cof[axis]
        if self._use_nugget:
            return self._marginal_model(h, *cof)
        return self._marginal_model(h, *cof, 0)

    def fit(self, force=False):
        """Fit the space-time model

        Fits the marginal model to both marginal variograms first. The sum
        and product models are then fully defined, the product-sum model
        fits its three parameters k1, k2 and k3 to the experimental
        variogram.

        Parameters
        ----------
        force : bool
            If True, an existing fit will be replaced.

        """
        if self.cof is not None and not force:
            return

        # fit the marginal models
        for axis in ('space', 'time'):
            cof, cov = self._fit_marginal(axis)
            self._marginal_cof[axis] = cof
            self._marginal_cov[axis] = cov

        # the marginal sills are the sill plus nugget
        Cx = self._marginal_cof['space'][1]
        Ct = self._marginal_cof['time'][1]
        if self._use_nugget:
            Cx += self._marginal_cof['space'][-1]
            Ct += self._marginal_cof['time'][-1]

        # the sum model does not use the marginal sills
        if self._model_name == 'sum':
            self._model_params = {}
        else:
            self._model_params = dict(Cx=Cx, Ct=Ct)

        # sum and product model have no fitting parameters
        if self._model_name != 'product-sum':
            self.cof = np.array([])
            self.cov = np.empty((0, 0))
            return

        def Vx(h):
            return self._marginal('space', h)

        def Vt(t):
            return self._marginal('time', t)

        # fit k1, k2, k3 to all filled lag classes
        xx, tt = self.meshbins
        idx = np.isfinite(self._experimental)
        lags = np.column_stack((xx[idx], tt[idx]))

        def wrapped(lags, k1, k2, k3):
            return self._model(lags, Vx, Vt, k1, k2, k3, Cx=Cx, Ct=Ct)

        self.cof, self.cov = curve_fit(
            wrapped,
            lags,
            self._experimental[idx],
            p0=np.ones(3),
            bounds=(0, np.inf)
        )

    @property
    def fitted_model(self):
        """Fitted space-time model

        Returns
        -------
        model : Callable
            Function that takes an array of shape (n, 2) holding space and
            time lag pairs and returns the modeled semi-variances.

        """
        # if the model is not fitted, fit it
        if self.cof is None:
            self.fit()

        func = self._model
        cof = self.cof
        params = self._model_params

        def Vx(h):
            return self._marginal('space', h)

        def Vt(t):
            return self._marginal('time', t)

        def model(lags):
            return func(np.asarray(lags, dtype=float), Vx, Vt, *cof, **params)

        return model

    def transform(self, lags):
        """Apply the fitted model

        Parameters
        ----------
        lags : numpy.ndarray
            Array of shape (n, 2) holding space and time lag pairs. Time
            lags are given in the internal time unit, which is seconds for
            datetime-like input.

        Returns
        -------
        semivariance : numpy.ndarray

        """
        return self.fitted_model(lags)

    # ----------------------------------------------------------------------- #
    #                              PROPERTIES                                 #
    # ----------------------------------------------------------------------- #
    @property
    def coordinates(self):
        """Spatial coordinates of all observations"""
        return self._X

    @property
    def time(self):
        """Observation times, as seconds since ``time_origin`` for datetime input"""
        return self._time

    @property
    def time_origin(self):
        """Oldest timestamp for datetime input, None for numeric time"""
        return self._time_origin

    @property
    def time_unit(self):
        """Plotting unit inferred from datetime input, None for numeric time"""
        return self._time_unit

    @property
    def values(self):
        return self._values

    @property
    def estimator(self):
        return self._estimator

    @property
    def xdistance(self):
        """Pairwise space lags"""
        return self._xdist

    @property
    def tdistance(self):
        """Pairwise time lags"""
        return self._tdist

    @property
    def xbins(self):
        """Upper bin edges of the space lag classes"""
        return self._xbins

    @property
    def tbins(self):
        """Upper bin edges of the time lag classes"""
        return self._tbins

    @property
    def x_lags(self):
        return len(self._xbins)

    @property
    def t_lags(self):
        return len(self._tbins)

    @property
    def meshbins(self):
        """Space and time bin edges as meshgrid of shape (x_lags, t_lags)"""
        return np.meshgrid(self._xbins, self._tbins, indexing='ij')

    @property
    def experimental(self):
        """Experimental variogram

        Returns
        -------
        variogram : numpy.ndarray
            Array of shape (x_lags, t_lags) holding the semi-variances.

        """
        return self._experimental

    @property
    def bin_count(self):
        """Number of point pairs in each lag class, shape (x_lags, t_lags)"""
        return self._bin_count

    # ----------------------------------------------------------------------- #
    #                             PLOTTING                                    #
    # ----------------------------------------------------------------------- #
    def plot(self, kind='scatter', ax=None, **kwargs):  # pragma: no cover
        """Plot the variogram

        Plots the experimental variogram and, once fitted, the space-time
        model. The time axis is shown in the unit implied by the data
        resolution.

        Parameters
        ----------
        kind : str
            Has to be one of:

            * **scatter**: 3D scatter of the experimental variogram
            * **surface**: 3D surface of the experimental variogram
            * **contour**: contour lines
            * **contourf**: filled contours
            * **matrix**: the experimental lag classes as image

        ax : matplotlib.Axes, plotly.graph_objects.Figure, None
            If None, a new figure is created. Else, the plot is added to
            the given object, which has to match the plotting backend.
        kwargs : dict
            Passed down to the plotting function. Set ``show_model=True``
            to fit the model on demand, ``show_model=False`` to only plot
            the experimental variogram.

        Returns
        -------
        fig : matplotlib.Figure, plotly.graph_objects.Figure

        """
        if not isinstance(kind, str):
            raise ValueError('kind has to be of type string.')
        kind = kind.lower()
        if kind not in ('scatter', 'surface', 'contour', 'contourf', 'matrix'):
            raise ValueError('%s is not a valid plot kind.' % kind)

        # backend
        used_backend = plotting.backend()
        if used_backend == 'matplotlib':
            return plotting.matplotlib_spatiotemporal_plot(self, kind=kind, ax=ax, **kwargs)
        elif used_backend == 'plotly':
            return plotting.plotly_spatiotemporal_plot(self, kind=kind, fig=ax, **kwargs)

        # if we reach this line, somethings wrong with plotting backend
        raise ValueError('The plotting backend has an undefined state.')
