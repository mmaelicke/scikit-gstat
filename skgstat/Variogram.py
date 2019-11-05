"""
Variogram class
"""
import copy

import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from numba import jit

from skgstat import estimators, models, binning


class Variogram(object):
    """Variogram Class

    Calculates a variogram of the separating distances in the given
    coordinates and relates them to one of the semi-variance measures of
    the given dependent values.

    """
    def __init__(self,
                 coordinates=None,
                 values=None,
                 estimator='matheron',
                 model='spherical',
                 dist_func='euclidean',
                 bin_func='even',
                 normalize=True,
                 fit_method='trf',
                 fit_sigma=None,
                 use_nugget=False,
                 maxlag=None,
                 n_lags=10,
                 verbose=False,
                 harmonize=False
                 ):
        r"""Variogram Class

        Note: The directional variogram estimation is not re-implemented yet.
        Therefore the parameters is-directional, azimuth and tolerance will
        be ignored at the moment and can be subject to changes.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Array of shape (m, n). Will be used as m observation points of
            n-dimensions. This variogram can be calculated on 1 - n
            dimensional coordinates. In case a 1-dimensional array is passed,
            a second array of same length containing only zeros will be
            stacked to the passed one.
        values : numpy.ndarray
            Array of values observed at the given coordinates. The length of
            the values array has to match the m dimension of the coordinates
            array. Will be used to calculate the dependent variable of the
            variogram.
        estimator : str, callable
            String identifying the semi-variance estimator to be used.
            Defaults to the Matheron estimator. Possible values are:

              * matheron        [Matheron, default]
              * cressie         [Cressie-Hawkins]
              * dowd            [Dowd-Estimator]
              * genton          [Genton]
              * minmax          [MinMax Scaler]
              * entropy         [Shannon Entropy]

            If a callable is passed, it has to accept an array of absoulte
            differences, aligned to the 1D distance matrix (flattened upper
            triangle) and return a scalar, that converges towards small
            values for similarity (high covariance).
        model : str
            String identifying the theoretical variogram function to be used
            to describe the experimental variogram. Can be one of:

              * spherical       [Spherical, default]
              * exponential     [Exponential]
              * gaussian        [Gaussian]
              * cubic           [Cubic]
              * stable          [Stable model]
              * matern          [Matérn model]
              * nugget          [nugget effect variogram]

        dist_func : str
            String identifying the distance function. Defaults to
            'euclidean'. Can be any metric accepted by
            scipy.spatial.distance.pdist. Additional parameters are not (yet)
            passed through to pdist. These are accepted by pdist for some of
            the metrics. In these cases the default values are used.
        bin_func : str
            String identifying the binning function used to find lag class
            edges. At the moment there are two possible values: 'even'
            (default) or 'uniform'. Even will find n_lags bins of same width
            in the interval [0,maxlag[. 'uniform' will identfy n_lags bins on
            the same interval, but with varying edges so that all bins count
            the same amount of observations.
        normalize : bool
            Defaults to False. If True, the independent and dependent
            variable will be normalized to the range [0,1].
        fit_method : str
            String identifying the method to be used for fitting the
            theoretical variogram function to the experimental. More info is
            given in the Variogram.fit docs. Can be one of:

                * 'lm': Levenberg-Marquardt algorithm for unconstrained
                  problems. This is the faster algorithm, yet is the fitting of
                  a variogram not unconstrianed.
                * 'trf': Trust Region Reflective function for non-linear
                  constrained problems. The class will set the boundaries
                  itself. This is the default function.

        fit_sigma : numpy.ndarray, str
            Defaults to None. The sigma is used as measure of uncertainty
            during variogram fit. If fit_sigma is an array, it has to hold
            n_lags elements, giving the uncertainty for all lags classes. If
            fit_sigma is None (default), it will give no weight to any lag.
            Higher values indicate higher uncertainty and will lower the
            influcence of the corresponding lag class for the fit.
            If fit_sigma is a string, a pre-defined function of separating
            distance will be used to fill the array. Can be one of:

                * 'linear': Linear loss with distance. Small bins will have
                  higher impact.
                * 'exp': The weights decrease by a e-function of distance
                * 'sqrt': The weights decrease by the squareroot of distance
                * 'sq': The weights decrease by the squared distance.

            More info is given in the Variogram.fit_sigma documentation.
        use_nugget : bool
            Defaults to False. If True, a nugget effet will be added to all
            Variogram.models as a third (or fourth) fitting parameter. A
            nugget is essentially the y-axis interception of the theoretical
            variogram function.
        maxlag : float, str
            Can specify the maximum lag distance directly by giving a value
            larger than 1. The binning function will not find any lag class
            with an edge larger than maxlag. If 0 < maxlag < 1, then maxlag
            is relative and maxlag * max(Variogram.distance) will be used.
            In case maxlag is a string it has to be one of 'median', 'mean'.
            Then the median or mean of all Variogram.distance will be used.
            Note maxlag=0.5 will use half the maximum separating distance,
            this is not the same as 'median', which is the median of all
            separating distances
        n_lags : int
            Specify the number of lag classes to be defined by the binning
            function.
        verbose : bool
            Set the Verbosity of the class. Not Implemented yet.
        harmonize : bool
            this kind of works so far, but will be rewritten (and documented)

        """
        # Set coordinates
        self._X = np.asarray(coordinates)

        # pairwise differences
        self._diff = None

        # set verbosity
        self.verbose = verbose

        # set values
        self._values = None
        self.set_values(values=values)

        # distance matrix
        self._dist = None

        # set distance calculation function
        self._dist_func = None
        self.set_dist_function(func=dist_func)

        # lags and max lag
        self._n_lags = None
        self.n_lags = n_lags
        self._maxlag = None
        self.maxlag = maxlag

        # estimator can be a function or a string
        self._estimator = None
        self.set_estimator(estimator_name=estimator)

        # model can be a function or a string
        self._model = None
        self.set_model(model_name=model)

        # the binning settings
        self._bin_func = None
        self._groups = None
        self._bins = None
        self.set_bin_func(bin_func=bin_func)

        # specify if the lag should be given absolute or relative to the maxlag
        self._normalized = normalize

        # specify if the experimental variogram shall be harmonized
        self.harmonize = harmonize

        # set if nugget effect shall be used
        self._use_nugget = None
        self.use_nugget = use_nugget

        # set the fitting method and sigma array
        self.fit_method = fit_method
        self._fit_sigma = None
        self.fit_sigma = fit_sigma

        # set attributes to be filled during calculation
        self.cov = None
        self.cof = None

        # settings, not reachable by init (not yet)
        self._cache_experimental = False

        # do the preprocessing and fitting upon initialization
        self.preprocessing(force=True)
        self.fit(force=True)

    @property
    def coordinates(self):
        """Coordinates property

        Array of observation locations the variogram is build for. This
        property has no setter. If you want to change the coordinates,
        use a new Variogram instance.

        Returns
        -------
        coordinates : numpy.array

        """
        return self._X

    @property
    def values(self):
        """Values property

        Array of observations, the variogram is build for. The setter of this
        property utilizes the Variogram.set_values function for setting new
        arrays.

        Returns
        -------
        values : numpy.ndarray

        See Also
        --------
        Variogram.set_values

        """
        return self._values

    @values.setter
    def values(self, values):
        self.set_values(values=values)

    @property
    def value_matrix(self):
        """Value matrix

        Returns a matrix of pairwise differences in absolute values. The
        matrix will have the shape (m, m) with m = len(Variogram.values).
        Note that Variogram.values holds the values themselves, while the
        value_matrix consists of their pairwise differences.

        Returns
        -------
        values : numpy.matrix
            Matrix of pairwise absolute differences of the values.

        See Also
        --------
        Variogram._diff

        """
        return squareform(self._diff)

    def set_values(self, values):
        """Set new values

        Will set the passed array as new value array. This array has to be of
        same length as the first axis of the coordinates array. The Variogram
        class does only accept one dimensional arrays.
        On success all fitting parameters are deleted and the pairwise
        differences are recalculated.

        Parameters
        ----------
        values : numpy.ndarray

        Returns
        -------
        void

        See Also
        --------
        Variogram.values

        """
        # check dimensions
        if not len(values) == len(self._X):
            raise ValueError('The length of the values array has to match' +
                             'the length of coordinates')

        # reset fitting parameter
        self.cof, self.cov = None, None
        self._diff = None

        # use an array
        _y = np.asarray(values)
        if not _y.ndim == 1:
            raise ValueError('The values shall be a 1-D array.' +
                             'Multi-dimensional values not supported yet.')

        # set new values
        self._values = np.asarray(values)

        # recalculate the pairwise differences
        self._calc_diff(force=True)

    @property
    def bin_func(self):
        """Binning function

        Returns an instance of the function used for binning the separating
        distances into the given amount of bins. Both functions use the same
        signature of func(distances, n, maxlag).

        The setter of this property utilizes the Variogram.set_bin_func to
        set a new function.

        Returns
        -------
        binning_function : function

        See Also
        --------
        Variogram.set_bin_func

        """
        return self._bin_func

    @bin_func.setter
    def bin_func(self, bin_func):
        self.set_bin_func(bin_func=bin_func)

    def set_bin_func(self, bin_func):
        """Set binning function

        Sets a new binning function to be used. The new binning method is set
        by a string identifying the new function to be used. Can be one of:
        ['even', 'uniform'].

        Parameters
        ----------
        bin_func : str
            Can be one of:

            * **'even'**: Use skgstat.binning.even_width_lags for using
              n_lags lags of equal width up to maxlag.
            * **'uniform'**: Use skgstat.binning.uniform_count_lags for using
              n_lags lags up to maxlag in which the pairwise differences
              follow a uniform distribution.

        Returns
        -------
        void

        See Also
        --------
        Variogram.bin_func
        skgstat.binning.uniform_count_lags
        skgstat.binning.even_width_lags

        """
        # switch the input
        if bin_func.lower() == 'even':
            self._bin_func = binning.even_width_lags
        elif bin_func.lower() == 'uniform':
            self._bin_func = binning.uniform_count_lags
        else:
            raise ValueError('%s binning method is not known' % bin_func)

        # reset groups and bins
        self._groups = None
        self._bins = None
        self.cof, self.cov = None, None

    @property
    def normalized(self):
        return self._normalized

    @normalized.setter
    def normalized(self, status):
        # set the new value
        self._normalized = status

    @property
    def bins(self):
        if self._bins is None:
            self._bins = self.bin_func(self.distance, self.n_lags, self.maxlag)

        return self._bins.copy()

    @bins.setter
    def bins(self, bins):
        # set the new bins
        self._bins = bins

        # clean the groups as they are not valid anymore
        self._groups = None
        self.cov = None
        self.cof = None

    @property
    def n_lags(self):
        return self._n_lags

    @n_lags.setter
    def n_lags(self, n):
        # TODO: here accept strings and implement some optimum methods
        if isinstance(n, str):
            raise NotImplementedError('n_lags string values not implemented')
        if not isinstance(n, int) or n < 1:
            raise ValueError('n_lags has to be a positive integer')
        self._n_lags = n

        # reset the bins and fitting
        self._bins = None
        self.cof = None
        self.cov = None

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

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self.set_model(model_name=value)

    def set_model(self, model_name):
        """
        Set model as the new theoretical variogram function.

        """
        # reset the fitting
        self.cof, self.cov = None, None

        if isinstance(model_name, str):
            if model_name.lower() == 'spherical':
                self._model = models.spherical
            elif model_name.lower() == 'exponential':
                self._model = models.exponential
            elif model_name.lower() == 'gaussian':
                self._model = models.gaussian
            elif model_name.lower() == 'cubic':
                self._model = models.cubic
            elif model_name.lower() == 'stable':
                self._model = models.stable
            elif model_name.lower() == 'matern':
                self._model = models.matern
            else:
                raise ValueError(
                    'The theoretical Variogram function %s is not' +
                    'understood, please provide the function' % model_name)
        else:
            self._model = model_name

    @property
    def use_nugget(self):
        return self._use_nugget

    @use_nugget.setter
    def use_nugget(self, nugget):
        if not isinstance(nugget, bool):
            raise ValueError('use_nugget has to be of type bool.')

        # set new value
        self._use_nugget = nugget

    @property
    def dist_function(self):
        return self._dist_func

    @dist_function.setter
    def dist_function(self, func):
        self.set_dist_function(func=func)

    def set_dist_function(self, func):
        """Set distance function

        Set the function used for distance calculation. func can either be a
        callable or a string. The ranked distance function is not implemented
        yet. strings will be forwarded to the scipy.spatial.distance.pdist
        function as the metric argument.
        If func is a callable, it has to return the upper triangle of the
        distance matrix as a flat array (Like the pdist function).

        Parameters
        ----------
        func : string, callable

        Returns
        -------
        numpy.array

        """
        # reset the distances and fitting
        self._dist = None
        self.cof, self.cov = None, None

        if isinstance(func, str):
            if func.lower() == 'rank':
                raise NotImplementedError
            else:
                # if not ranks, it has to be a scipy metric
                self._dist_func = lambda x: pdist(X=x, metric=func)

        elif callable(func):
            self._dist_func = func
        else:
            raise ValueError('Input not supported. Pass a string or callable.')

        # re-calculate distances
        self._calc_distances()

    @property
    def distance(self):
        if self._dist is None:
            self._calc_distances()
        return self._dist

    @distance.setter
    def distance(self, dist_array):
        self._dist = dist_array

    @property
    def distance_matrix(self):
        return squareform(self.distance)

    @property
    def maxlag(self):
        return self._maxlag

    @maxlag.setter
    def maxlag(self, value):
        # reset fitting
        self.cof, self.cov = None, None

        # remove bins
        self._bins = None
        self._groups = None

        # set new maxlag
        if value is None:
            self._maxlag = None
        elif isinstance(value, str):
            if value == 'median':
                self._maxlag = np.median(self.distance)
            elif value == 'mean':
                self._maxlag = np.mean(self.distance)
        elif value < 1:
            self._maxlag = value * np.max(self.distance)
        else:
            self._maxlag = value

    @property
    def fit_sigma(self):
        r"""Fitting Uncertainty

        Set or calculate an array of observation uncertainties aligned to the
        Variogram.bins. These will be used to weight the observations in the
        cost function, which divides the residuals by their uncertainty.

        When setting fit_sigma, the array of uncertainties itself can be
        given, or one of the strings: ['linear', 'exp', 'sqrt', 'sq']. The
        parameters described below refer to the setter of this property.

        Parameters
        ----------
        sigma : string, array
            Sigma can either be an array of discrete uncertainty values,
            which have to align to the Variogram.bins, or of type string.
            Then, the weights for fitting are calculated as a function of
            (lag) distance.

              * **sigma='linear'**: The residuals get weighted by the lag
                distance normalized to the maximum lag distance, denoted as
                :math:`w_n`
              * **sigma='exp'**: The residuals get weighted by the function:
                :math:`w = e^{1 / w_n}`
              * **sigma='sqrt'**: The residuals get weighted by the function:
                :math:`w = \sqrt(w_n)`
              * **sigma='sq'**: The residuals get weighted by the function:
                :math:`w = w_n^2`

        Returns
        -------
        void


        Notes
        -----
        The cost function is defined as:

        .. math::
            chisq = \sum {\frac{r}{\sigma}}^2

        where r are the residuals between the experimental variogram and the
        modeled values for the same lag. Following this function,
        small values will increase the influence of that residual, while a very
        large sigma will cause the observation to be ignored.

        See Also
        --------
        scipy.optimize.curve_fit

        """
        # unweighted
        if self._fit_sigma is None:
            return None

        # discrete uncertainties
        elif isinstance(self._fit_sigma, (list, tuple, np.ndarray)):
            if len(self._fit_sigma) == len(self._bins):
                return self._fit_sigma
            else:
                raise AttributeError('fit_sigma and bins need the same length.')

        # linear function of distance
        elif self._fit_sigma == 'linear':
            return self.bins / np.max(self.bins)

        # e function of distance
        elif self._fit_sigma == 'exp':
            return 1. / np.exp(1. / (self.bins / np.max(self.bins)))

        # sqrt function of distance
        elif self._fit_sigma == 'sqrt':
            return np.sqrt(self.bins / np.max(self.bins))

        # squared function of distance
        elif self._fit_sigma == 'sq':
            return (self.bins / np.max(self.bins)) ** 2
        else:
            raise ValueError("fit_sigma is not understood. It has to be an " +
                             "array or one of ['linear', 'exp', 'sqrt', 'sq'].")

    @fit_sigma.setter
    def fit_sigma(self, sigma):
        self._fit_sigma = sigma

        # remove fitting parameters
        self.cof = None
        self.cov = None

    def lag_groups(self):
        """Lag class groups

        Retuns a mask array with as many elements as self._diff has,
        identifying the lag class group for each pairwise difference. Can be
        used to extract all pairwise values within the same lag bin.

        Returns
        -------
        numpy.ndarray

        See Also
        --------
        Variogram.lag_classes

        """
        if self._groups is None:
            self._calc_groups()

        return self._groups

    def lag_classes(self):
        """Iterate over the lag classes

        Generates an iterator over all lag classes. Can be zipped with
        Variogram.bins to identify the lag.

        Returns
        -------
        iterable

        """
        # yield all groups
        for i in np.unique(self.lag_groups()):
            if i < 0:
                continue
            else:
                yield self._diff[np.where(self.lag_groups() == i)]

    def preprocessing(self, force=False):
        """Preprocessing function

        Prepares all input data for the fit and transform functions. Namely,
        the distances are calculated and the value differences. Then the
        binning is set up and bin edges are calculated. If any of the listed
        subsets are already prepared, their processing is skipped. This
        behaviour can be changed by the force parameter. This will cause a
        clean preprocessing.

        Parameters
        ----------
        force : bool
            If set to True, all preprocessing data sets will be deleted. Use
            it in case you need a clean preprocessing.

        Returns
        -------
        void

        """
        # call the _calc functions
        self._calc_distances(force=force)
        self._calc_diff(force=force)
        self._calc_groups(force=force)

    def fit(self, force=False, method=None, sigma=None, **kwargs):
        """Fit the variogram

        The fit function will fit the theoretical variogram function to the
        experimental. The preprocessed distance matrix, pairwise differences
        and binning will not be recalculated, if already done. This could be
        forced by setting the force parameter to true.

        In case you call fit function directly, with method or sigma,
        the parameters set on Variogram object instantiation will get
        overwritten. All other keyword arguments will be passed to
        scipy.optimize.curve_fit function.

        Parameters
        ----------
        force : bool
            If set to True, a clean preprocessing of the distance matrix,
            pairwise differences and the binning will be forced. Default is
            False.
        method : string
            A string identifying one of the implemented fitting procedures.
            Can be one of ['lm', 'trf']:

              * lm: Levenberg-Marquardt algorithms implemented in
                scipy.optimize.leastsq function.
              * trf: Trust Region Reflective algorithm implemented in
                scipy.optimize.least_squares(method='trf')

        sigma : string, array
            Uncertainty array for the bins. Has to have the same dimension as
            self.bins. Refer to Variogram.fit_sigma for more information.

        Returns
        -------
        void

        See Also
        --------
        scipy.optimize
        scipy.optimize.curve_fit
        scipy.optimize.leastsq
        scipy.optimize.least_squares

        """
        # TODO: the kwargs need to be preserved somehow
        # delete the last cov and cof
        self.cof = None
        self.cov = None

        # if force, force a clean preprocessing
        self.preprocessing(force=force)

        # load the data
        x = self.bins
        y = self.experimental

        # overwrite fit setting if new params are given
        if method is not None:
            self.fit_method = method
        if sigma is not None:
            self.fit_sigma = sigma

        # remove nans
        _x = x[~np.isnan(y)]
        _y = y[~np.isnan(y)]

        # the model function is re-defined. otherwise scipy cannot determine
        # the number of parameters
        # TODO: def f(n of par)

        # Switch the method
        # Trust Region Reflective
        if self.fit_method == 'trf':
            bounds = (0, self.__get_fit_bounds(x, y))
            self.cof, self.cov = curve_fit(
                self._model,
                _x, _y,
                method='trf',
                sigma=self.fit_sigma,
                p0=bounds[1],
                bounds=bounds,
                **kwargs
            )

        # Levenberg-Marquardt
        elif self.fit_method == 'lm':
            self.cof, self.cov = curve_fit(
                self.model,
                _x, _y,
                method='lm',
                sigma=self.fit_sigma,
                **kwargs
            )

        else:
            raise ValueError('Only the \'lm\' and \'trf\' algorithms are ' +
                             'supported at the moment.')

    def transform(self, x):
        """Transform

        Transform a given set of lag values to the theoretical variogram
        function using the actual fitting and preprocessing parameters in
        this Variogram instance

        Parameters
        ----------
        x : numpy.array
            Array of lag values to be used as model input for the fitted
            theoretical variogram model

        Returns
        -------
        numpy.array

        """
        self.preprocessing()

        # if instance is not fitted, fit it
        if self.cof is None:
            self.fit(force=True)

        return np.fromiter(map(self.compiled_model, x), dtype=float)

    @property
    def compiled_model(self):
        """Compiled theoretical variogram model

        Compile the model using the actual fitting parameters to return a
        function implementing them.

        Deprecated
        ----------
        The compiled_model will be removed in version 0.3. Use the
        `Variogram.fitted_model` property instead. It is works in the same
        way, but is significantly faster

        Returns
        -------
        callable

        """
        if self.cof is None:
            self.fit(force=True)

        # get the function
        func = self._model

        # get the pars
        params = self.describe()
        ming = params['nugget']
        maxg = params['sill']

        # define the wrapper
        def model(x):
            gamma = func(x, *self.cof)
            if int(x) == 0 and not np.isfinite(gamma):
                return ming
            elif int(x) > 0 and not np.isfinite(gamma):
                return maxg
            else:
                return gamma

        # return
        return model

    @property
    def fitted_model(self):
        """Fitted Model

        Returns a callable that takes a distance value and returns a
        semivariance. This model is fitted to the current Variogram
        parameters. The function will be interpreted at return time with the
        parameters hard-coded into the function code. This makes it way
        faster than`Variogram.compiled_model`.

        Returns
        -------
        model : callable
            The current semivariance model fitted to the current Variogram
            model parameters.
        """
        if self.cof is None:
            self.fit(force=True)

        # get the pars
        cof = self.cof

        # get the function
        func = self._model

        code = """model = lambda x: func(x, %s)""" % \
               (', '.join([str(_) for _ in cof]))

        # run the code
        loc = dict(func=func)
        exec(code, loc, loc)
        model = loc['model']

        return model

    def _calc_distances(self, force=False):
        if self._dist is not None and not force:
            return

        # if self._X is of just one dimension, concat zeros.
        if self._X.ndim == 1:
            _x = np.vstack(zip(self._X, np.zeros(len(self._X))))
        else:
            _x = self._X
        # else calculate the distances
        self._dist = self._dist_func(_x)

    def _calc_diff(self, force=False):
        """Calculates the pairwise differences

        Returns
        -------
        void

        """
        # already calculated
        if self._diff is not None and not force:
            return

        v = self.values
        l = len(v)
        self._diff = np.zeros(int((l**2 - l) / 2))

        # calculate the pairwise differences
        for t, k in zip(self.__vdiff_indexer(), range(len(self._diff))):
            self._diff[k] = np.abs(v[t[0]] - v[t[1]])

    #@jit
    def __vdiff_indexer(self):
        """Pairwise indexer

        Returns an iterator over the values or coordinates in squareform
        coordinates. The iterable will be of type tuple.

        Returns
        -------
        iterable

        """
        l = len(self.values)

        for i in range(l):
            for j in range(l):
                if i < j:
                    yield i, j

    def _calc_groups(self, force=False):
        """Calculate the lag class mask array

        Returns
        -------

        """
        # already calculated
        if self._groups is not None and not force:
            return

        # get the bin edges and distances
        bin_edges = self.bins
        d = self.distance

        # -1 is the group fir distances outside maxlag
        self._groups = np.ones(len(d), dtype=int) * -1

        for i, bounds in enumerate(zip([0] + list(bin_edges), bin_edges)):
            self._groups[np.where((d >= bounds[0]) & (d < bounds[1]))] = i

    def clone(self):
        """Deep copy of self

        Return a deep copy of self.

        Returns
        -------
        Variogram

        """
        return copy.deepcopy(self)

    @property
    def experimental(self):
        """Experimental Variogram

        Array of experimental (empirical) semivariance values. The array
        length will be aligned to Variogram.bins. The current
        Variogram.estimator has been used to calculate the values. Depending
        on the setting of Variogram.harmonize (True | False), either
        Variogram._experimental or Variogram.isotonic will be returned.

        Returns
        -------
        vario : numpy.ndarray
            Array of the experimental semi-variance values aligned to
            Variogram.bins.

        See Also
        --------
        Variogram._experimental
        Variogram.isotonic

        """
        if self.harmonize:
            return self.isotonic
        else:
            return self._experimental

    @property
#    @jit
    def _experimental(self):
        """

        Returns
        -------

        """
        # prepare the result array
        y = np.zeros(len(self.bins), dtype=np.float64)

        # args, can set the bins for entropy
        # and should set p of percentile, not properly implemented
        if self._estimator.__name__ == 'entropy':
            bins = np.linspace(
                np.min(self.distance),
                np.max(self.distance),
                50
            )
            # apply
            for i, lag_values in enumerate(self.lag_classes()):
                y[i] = self._estimator(lag_values, bins=bins)

        # default
        else:
            for i, lag_values in enumerate(self.lag_classes()):
                y[i] = self._estimator(lag_values)

        # apply
        return y.copy()

    @property
    def isotonic(self):
        """
        Return the isotonic harmonized experimental variogram.
        This means, the experimental variogram is monotonic after harmonization.

        The harmonization is done using PAVA algorithm:

        Barlow, R., D. Bartholomew, et al. (1972): Statistical Interference Under Order Restrictions.
            John Wiley and Sons, New York.
        Hiterding, A. (2003): Entwicklung hybrider Interpolationsverfahren für den automatisierten Betrieb am
            Beispiel meteorologischer Größen. Dissertation, Institut für Geoinformatik, Westphälische
            Wilhelms-Universität Münster, IfGIprints, Münster. ISBN: 3-936616-12-4

        TODO: solve the import

        :return: np.ndarray, monotonized experimental variogram
        """
        # TODO this is imported in the function as sklearn is not a dependency (and should not be for now)
        raise NotImplementedError

        try:
            from sklearn.isotonic import IsotonicRegression
            y = self._experimental
            x = self.bins
            return IsotonicRegression().fit_transform(x,y)
        except ImportError:
            raise NotImplementedError('scikit-learn is not installed, but the isotonic function without sklear dependency is not installed yet.')

    def __get_fit_bounds(self, x, y):
        """
        Return the bounds for parameter space in fitting a variogram model.
        The bounds are depended on the Model that is used

        Returns
        -------
        list

        """
        mname = self._model.__name__

        # use range, sill and smoothness parameter
        if mname == 'matern':
            # a is max(x), C0 is max(y) s is limited to 20?
            bounds = [np.nanmax(x), np.nanmax(y), 20.]

        # use range, sill and shape parameter
        elif mname == 'stable':
            # a is max(x), C0 is max(y) s is limited to 2?
            bounds = [np.nanmax(x), np.nanmax(y), 2.]

        # use only sill
        elif mname == 'nugget':
            # a is max(x):
            bounds = [np.nanmax(x)]

        # use range and sill
        else:
            # a is max(x), C0 is max(y)
            bounds = [np.nanmax(x), np.nanmax(y)]

        # if use_nugget is True add the nugget
        if self.use_nugget:
            bounds.append(0.99)

        return bounds

    def data(self, n=100, force=False):
        """Theoretical variogram function

        Calculate the experimental variogram and apply the binning. On
        success, the variogram model will be fitted and applied to n lag
        values. Returns the lags and the calculated semi-variance values.
        If force is True, a clean preprocessing and fitting run will be
        executed.

        Parameters
        ----------
        n : integer
            length of the lags array to be used for fitting. Defaults to 100,
            which will be fine for most plots
        force: boolean
            If True, the preprocessing and fitting will be executed as a
            clean run. This will force all intermediate results to be
            recalculated. Defaults to False

        Returns
        -------
        variogram : tuple
            first element is the created lags array
            second element are the calculated semi-variance values

        """
        # force a clean preprocessing if needed
        self.preprocessing(force=force)

        # calculate the experimental variogram
        _exp = self.experimental
        _bin = self.bins

        # use relative or absolute bins
        if self.normalized:
            _bin /= np.nanmax(_bin)     # normalize X
            _exp /= np.nanmax(_exp)     # normalize Y
            x = np.linspace(0, 1, n)    # use n increments
        else:
            x = np.linspace(0, np.float64(np.nanmax(_bin)), n)

        # fit if needed
        if self.cof is None:
            self.fit(force=force)

        return x, self._model(x, *self.cof)

    @property
    def residuals(self):
        """Model residuals

        Calculate the model residuals defined as the differences between the
        experimental variogram and the theoretical model values at
        corresponding lag values

        Returns
        -------
        numpy.ndarray

        """
        # get the deviations
        experimental, model = self.model_deviations()

        # calculate the residuals
        return np.fromiter(
            map(lambda x, y: x - y, model, experimental),
            np.float
        )

    @property
    def mean_residual(self):
        """Mean Model residuals

        Calculates the mean, absoulte deviations between the experimental
        variogram and theretical model values.

        Returns
        -------
        float
        """
        return np.nanmean(np.fromiter(map(np.abs, self.residuals), np.float))

    @property
    def rmse(self):
        r"""RMSE

        Calculate the Root Mean squared error between the experimental
        variogram and the theoretical model values at corresponding lags.
        Can be used as a fitting quality measure.

        Returns
        -------
        float

        See Also
        --------
        Variogram.residuals

        Notes
        -----
        The RMSE is implemented like:

        .. math::
            RMSE = \sqrt{\frac{\sum_{i=0}^{i=N(x)} (x-y)^2}{N(x)}}

        """
        # get the deviations
        experimental, model = self.model_deviations()

        # get the sum of squares
        rsum = np.nansum(np.fromiter(
            map(lambda x, y: (x - y)**2, experimental, model),
            np.float
        ))

        return np.sqrt(rsum / len(model))

    @property
    def nrmse(self):
        r"""NRMSE

        Calculate the normalized root mean squared error between the
        experimental variogram and the theoretical model values at
        corresponding lags. Can be used as a fitting quality measure

        Returns
        -------
        float

        See Also
        --------
        Variogram.residuals
        Variogram.rmse

        Notes
        -----

        The NRMSE is implemented as:

        .. math::

            NRMSE = \frac{RMSE}{mean(y)}

        where RMSE is Variogram.rmse and y is Variogram.experimental


        """
        return self.rmse / np.nanmean(self.experimental)

    @property
    def nrmse_r(self):
        r"""NRMSE

        Alternative normalized root mean squared error between the
        experimental variogram and the theoretical model values at
        corresponding lags. Can be used as a fitting quality measure.

        Returns
        -------
        float

        See Also
        --------
        Variogram.rmse
        Variogram.nrmse

        Notes
        -----
        Unlike Variogram.nrmse, nrmse_r is not normalized to the mean of y,
        but the differece of the maximum y to its mean:

        .. math::
            NRMSE_r = \frac{RMSE}{max(y) - mean(y)}

        """
        _y = self.experimental
        return self.rmse / (np.nanmax(_y) - np.nanmean(_y))

    @property
    def r(self):
        """
        Pearson correlation of the fitted Variogram

        :return:
        """
        # get the experimental and theoretical variogram and cacluate means
        experimental, model = self.model_deviations()
        mx = np.nanmean(experimental)
        my = np.nanmean(model)

        # claculate the single pearson correlation terms
        term1 = np.nansum(np.fromiter(map(lambda x, y: (x-mx) * (y-my), experimental, model), np.float))

        t2x = np.nansum(np.fromiter(map(lambda x: (x-mx)**2, experimental), np.float))
        t2y = np.nansum(np.fromiter(map(lambda y: (y-my)**2, model), np.float))

        return term1 / (np.sqrt(t2x * t2y))

    @property
    def NS(self):
        """
        Nash Sutcliffe efficiency of the fitted Variogram

        :return:
        """
        experimental, model = self.model_deviations()
        mx = np.nanmean(experimental)

        # calculate the single nash-sutcliffe terms
        term1 = np.nansum(np.fromiter(map(lambda x, y: (x - y)**2, experimental, model), np.float))
        term2 = np.nansum(np.fromiter(map(lambda x: (x - mx)**2, experimental), np.float))

        return 1 - (term1 / term2)

    def model_deviations(self):
        """Model Deviations

        Calculate the deviations between the experimental variogram and the
        recalculated values for the same bins using the fitted theoretical
        variogram function. Can be utilized to calculate a quality measure
        for the variogram fit.

        Returns
        -------
        deviations : tuple
            first element is the experimental variogram
            second element are the corresponding values of the theoretical
            model.

        """
        # get the experimental values and their bin bounds
        _exp = self.experimental
        _bin = self.bins

        # get the model parameters
        param = self.describe()
        if 'error' in param:
            raise RuntimeError('The variogram cannot be calculated.')

        # calculate the model values at bin bounds
        _model = self.transform(_bin)

        return _exp, _model

    def describe(self):
        """Variogram parameters

        Return a dictionary of the variogram parameters.

        Returns
        -------
        dict

        """
        # fit, if not already done
        if self.cof is None:
            self.fit(force=True)

        # scale sill and range
        if self.normalized:
            maxlag = np.nanmax(self.bins)
            maxvar = np.nanmax(self.experimental)
        else:
            maxlag = 1.
            maxvar = 1.

        # get the fitting coefficents
        cof = self.cof

        # build the dict
        rdict = dict(
            name=self._model.__name__,
            estimator=self._estimator.__name__,
            effective_range=cof[0] * maxlag,
            sill=cof[1] * maxvar,
            nugget=cof[-1] * maxvar if self.use_nugget else 0
        )

        # handle s parameters for matern and stable model
        if self._model.__name__ == 'matern':
            rdict['smoothness'] = cof[2]
        elif self._model.__name__ == 'stable':
            rdict['shape'] = cof[2]

        # return
        return rdict

    @property
    def parameters(self):
        """
        Extract just the variogram parameters range, sill and nugget from the self.describe return

        :return:
        """
        d = self.describe()
        if 'error' in d:
            return [None, None, None]
        elif self._model.__name__ == 'matern':
            return list([
                d['effective_range'],
                d['sill'],
                d['smoothness'],
                d['nugget']
            ])
        elif self._model.__name__ == 'stable':
            return list([
                d['effective_range'],
                d['sill'],
                d['shape'],
                d['nugget']
            ])
        elif self._model.__name__ == 'nugget':
            return list([d['nugget']])
        else:
            return list([
                d['effective_range'],
                d['sill'],
                d['nugget']
            ])

    def to_DataFrame(self, n=100, force=False):
        """Variogram DataFrame

        Returns the fitted theoretical variogram as a pandas.DataFrame
        instance. The n and force parameter control the calaculation,
        refer to the data funciton for more info.

        Parameters
        ----------
        n : integer
            length of the lags array to be used for fitting. Defaults to 100,
            which will be fine for most plots
        force: boolean
            If True, the preprocessing and fitting will be executed as a
            clean run. This will force all intermediate results to be
            recalculated. Defaults to False

        Returns
        -------
        pandas.DataFrame

        See Also
        --------
        Variogram.data

        """
        lags, data = self.data(n=n, force=force)

        return DataFrame({
            'lags': lags,
            self._model.__name__: data}
        ).copy()

    def plot(self, axes=None, grid=True, show=True, hist=True):
        """Variogram Plot

        Plot the experimental variogram, the fitted theoretical function and
        an histogram for the lag classes. The axes attribute can be used to
        pass a list of AxesSubplots or a single instance to the plot
        function. Then these Subplots will be used. If only a single instance
        is passed, the hist attribute will be ignored as only the variogram
        will be plotted anyway.

        Parameters
        ----------
        axes : list, tuple, array, AxesSubplot or None
            If None, the plot function will create a new matplotlib figure.
            Otherwise a single instance or a list of AxesSubplots can be
            passed to be used. If a single instance is passed, the hist
            attribute will be ignored.
        grid : bool
            Defaults to True. If True a custom grid will be drawn through
            the lag class centers
        show : bool
            Defaults to True. If True, the show method of the passed or
            created matplotlib Figure will be called before returning the
            Figure. This should be set to False, when used in a Notebook,
            as a returned Figure object will be plotted anyway.
        hist : bool
            Defaults to True. If False, the creation of a histogram for the
            lag classes will be suppressed.

        Returns
        -------
        matplotlib.Figure

        """
        # get the parameters
        _bins = self.bins
        _exp = self.experimental
        x = np.linspace(0, np.nanmax(_bins), 100)  # make the 100 a param?

        # do the plotting
        if axes is None:
            if hist:
                fig = plt.figure(figsize=(8, 5))
                ax1 = plt.subplot2grid((5, 1), (1, 0), rowspan=4)
                ax2 = plt.subplot2grid((5, 1), (0, 0), sharex=ax1)
                fig.subplots_adjust(hspace=0)
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
                ax2 = None
        elif isinstance(axes, (list, tuple, np.ndarray)):
            ax1, ax2 = axes
            fig = ax1.get_figure()
        else:
            ax1 = axes
            ax2 = None
            fig = ax1.get_figure()

        # apply the model
        y = self.transform(x)

        # handle the relative experimental variogram
        if self.normalized:
            _bins /= np.nanmax(_bins)
            y /= np.max(_exp)
            _exp /= np.nanmax(_exp)
            x /= np.nanmax(x)

        # ------------------------
        # plot Variograms
        ax1.plot(_bins, _exp, '.b')
        ax1.plot(x, y, '-g')

        # ax limits
        if self.normalized:
            ax1.set_xlim([0, 1.05])
            ax1.set_ylim([0, 1.05])
        if grid:
            ax1.grid(False)
            ax1.vlines(_bins, *ax1.axes.get_ybound(), colors=(.85, .85, .85),
                       linestyles='dashed')
        # annotation
        ax1.axes.set_ylabel('semivariance (%s)' % self._estimator.__name__)
        ax1.axes.set_xlabel('Lag (-)')

        # ------------------------
        # plot histogram
        if ax2 is not None and hist:
            # calc the histogram
            _count = np.fromiter(
                (g.size for g in self.lag_classes()), dtype=int
            )

            # set the sum of hist bar widths to 70% of the x-axis space
            w = (np.max(_bins) * 0.7) / len(_count)

            # plot
            ax2.bar(_bins, _count, width=w, align='center', color='red')

            # adjust
            plt.setp(ax2.axes.get_xticklabels(), visible=False)
            ax2.axes.set_yticks(ax2.axes.get_yticks()[1:])

            # need a grid?
            if grid:
                ax2.grid(False)
                ax2.vlines(_bins, *ax2.axes.get_ybound(),
                           colors=(.85, .85, .85), linestyles='dashed')

            # anotate
            ax2.axes.set_ylabel('N')

        # show the figure
        if show:
            fig.show()

        return fig

    def scattergram(self, ax=None):

        # create a new plot or use the given
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        tail = np.empty(0)
        head = tail.copy()

        for h in np.unique(self.lag_groups()):
            # get the head and tail
            x, y = np.where(squareform(self.lag_groups()) == h)

            # concatenate
            tail = np.concatenate((tail, self.values[x]))
            head = np.concatenate((head, self.values[y]))

        # plot the mean on tail and head
        ax.vlines(np.mean(tail), np.min(tail), np.max(tail), linestyles='--',
                  color='red', lw=2)
        ax.hlines(np.mean(head), np.min(head), np.max(head), linestyles='--',
                  color='red', lw=2)
        # plot
        ax.scatter(tail, head, 10, marker='o', color='orange')

        # annotate
        ax.set_ylabel('head')
        ax.set_xlabel('tail')

        # show the figure
        fig.show()

        return fig

    def location_trend(self, axes=None):
        """Location Trend plot

        Plots the values over each dimension of the coordinates in a scatter
        plot. This will visually show correlations between the values and any
        of the coordinate dimension. If there is a value dependence on the
        location, this would violate the intrinsic hypothesis. This is a
        weaker form of stationarity of second order.

        Parameters
        ----------
        axes : list
            Can be None (default) or a list of matplotlib.AxesSubplots. If a
            list is passed, the location trend plots will be plotted on the
            given instances. Note that then length of the list has to match
            the dimeonsionality of the coordinates array. In case 3D
            coordinates are used, three subplots have to be given.

        Returns
        -------
        matplotlib.Figure

        """
        N = len(self._X[0])
        if axes is None:
            # derive the needed amount of col and row
            nrow = int(round(np.sqrt(N)))
            ncol = int(np.ceil(N / nrow))
            fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 6 ,nrow * 6))
        else:
            if not len(axes) == N:
                raise ValueError(
                    'The amount of passed axes does not fit the coordinate' +
                    ' dimensionality of %d' % N)
            fig = axes[0].get_figure()

        for i in range(N):
            axes.flatten()[i].plot([_[i] for _ in self._X], self.values, '.r')
            axes.flatten()[i].set_xlabel('%d-dimension' % (i + 1))
            axes.flatten()[i].set_ylabel('value')

        # plot the figure and return it
        plt.tight_layout()
        fig.show()

        return fig

    def distance_difference_plot(self, ax=None, plot_bins=True, show=True):
        """Raw distance plot

        Plots all absoulte value differences of all point pair combinations
        over their separating distance, without sorting them into a lag.

        Parameters
        ----------
        ax : None, AxesSubplot
            If None, a new matplotlib.Figure will be created. In case a
            Figure was already created, pass the Subplot to use as ax argument.
        plot_bins : bool
            If True (default) the bin edges will be included into the plot.
        show : bool
            If True (default), the show method of the Figure will be called
            before returning the Figure. Can be set to False, to avoid
            doubled figure rendering in Jupyter notebooks.

        Returns
        -------
        matplotlib.pyplot.Figure

        """
        # get all distances
        _dist = self.distance

        # get all differences
        if self._diff is None:
            self._calc_diff()
        _diff = self._diff

        # create the plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        else:
            fig = ax.get_figure()

        # plot the bins
        if plot_bins:
            _bins = self.bins
            ax.vlines(_bins, 0, np.max(_diff), linestyle='--', lw=1, color='r')

        # plot
        ax.scatter(_dist, _diff, 8, color='b', marker='o', alpha=0.5)

        # set limits
        ax.set_ylim((0, np.max(_diff)))
        ax.set_xlim((0, np.max(_dist)))
        ax.set_xlabel('separating distance')
        ax.set_ylabel('pairwise difference')
        ax.set_title('Pairwise distance ~ difference')

        # show the plot
        if show:
            fig.show()

        return fig

    def __repr__(self):
        """
        Textual representation of this Variogram instance.

        :return:
        """
        try:
            _name = self._model.__name__
            _b = int(len(self.bins))
        except:
            return "< abstract Variogram >"
        return "< %s Semivariogram fitted to %d bins >" % (_name, _b)

    def __str__(self):
        """String Representation

        Descriptive respresentation of this Variogram instance that shall give
        the main variogram parameters in a print statement.

        Returns
        -------
        description : str
            String description of the variogram instance. Described by the
            Variogram parameters.

        """
        par = self.describe()

        _sill = np.NaN if 'error' in par else par['sill']
        _range = np.NaN if 'error' in par else par['effective_range']
        _nugget = np.NaN if 'error' in par else par['nugget']

        s = "{0} Variogram\n".format(par['name'])
        s+= "-" * (len(s) - 1) + "\n"
        s+="""Estimator:         %s
        \rEffective Range:   %.2f
        \rSill:              %.2f
        \rNugget:            %.2f
        """ % (par['estimator'], _range, _sill, _nugget)

        return s
