"""
Variogram class
"""
import copy
import inspect
import warnings
from typing import Iterable, Callable, Union, Tuple

import numpy as np
from pandas import DataFrame
from scipy.optimize import curve_fit, minimize, OptimizeWarning
from scipy.spatial.distance import pdist, squareform
from scipy import stats, sparse
from sklearn.isotonic import IsotonicRegression

from skgstat import estimators, models, binning
from skgstat import plotting
from skgstat.util import shannon_entropy
from .MetricSpace import MetricSpace, ProbabalisticMetricSpace
from skgstat.interfaces.gstools import skgstat_to_gstools, skgstat_to_krige


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
                 normalize=False,
                 fit_method='trf',
                 fit_sigma=None,
                 use_nugget=False,
                 maxlag=None,
                 samples=None,
                 n_lags=10,
                 verbose=False,
                 **kwargs,
                 ):
        r"""Variogram Class

        Parameters
        ----------
        coordinates : numpy.ndarray, MetricSpace
            .. versionchanged:: 0.5.0
                now accepts MetricSpace
            Array of shape (m, n). Will be used as m observation points of
            n-dimensions. This variogram can be calculated on 1 - n
            dimensional coordinates. In case a 1-dimensional array is passed,
            a second array of same length containing only zeros will be
            stacked to the passed one.
            For very large datasets, you can set maxlag to only calculate
            distances within the maximum lag in a sparse matrix.
            Alternatively you can supply a MetricSpace (optionally with a
            `max_dist` set for the same effect). This is useful if you're
            creating many different variograms for different measured
            parameters that are all measured at the same set of coordinates,
            as distances will only be calculated once, instead of once per
            variogram.
        values : numpy.ndarray
            .. versionchanged:: 1.0.5
                Now accepts co-variables for calculating cross variograms.

            Array of values observed at the given coordinates. The length of
            the values array has to match the m dimension of the coordinates
            array. Will be used to calculate the dependent variable of the
            variogram.
            If the values are of shape ``(n_samples, 2)``, a cross-variogram
            will be calculated. This assumes the main variable and the
            co-variable to be co-located under Markov-model 1 assumptions,
            meaning the variable need to be conditionally independent.
        estimator : str, callable
            String identifying the semi-variance estimator to be used.
            Defaults to the Matheron estimator. Possible values are:

              * matheron        [Matheron, default]
              * cressie         [Cressie-Hawkins]
              * dowd            [Dowd-Estimator]
              * genton          [Genton]
              * minmax          [MinMax Scaler]
              * entropy         [Shannon Entropy]

            If a callable is passed, it has to accept an array of absolute
            differences, aligned to the 1D distance matrix (flattened upper
            triangle) and return a scalar, that converges towards small
            values for similarity (high covariance).
        model : str | Callable
            .. versionchanged:: 1.0.12
                Added support for sum of models (e.g., "spherical+gaussian"), or custom model (Callable). Using
                `fit_bounds` to optimize the fit is recommended for custom models, and can be useful for sum of models.

            String or callable identifying the theoretical variogram function to be used
            to describe the experimental variogram. Can be one of:

              * spherical       [Spherical, default]
              * exponential     [Exponential]
              * gaussian        [Gaussian]
              * cubic           [Cubic]
              * stable          [Stable model]
              * matern          [Matérn model]
              * nugget          [nugget effect variogram]

            Any number of these theoretical models can be summed using "+" iteratively, e.g. "spherical+cubic+matern".
            The nugget parameters of the models are removed except for the last model (sum of nuggets = single nugget).

        dist_func : str
            String identifying the distance function. Defaults to
            'euclidean'. Can be any metric accepted by
            scipy.spatial.distance.pdist. Additional parameters are not (yet)
            passed through to pdist. These are accepted by pdist for some of
            the metrics. In these cases the default values are used.
        bin_func : str | Callable | Iterable
            .. versionchanged:: 0.3.8
                added 'fd', 'sturges', 'scott', 'sqrt', 'doane'
            .. versionchanged:: 0.3.9
                added 'kmeans', 'ward'

            String identifying the binning function used to find lag class
            edges. All methods calculate bin edges on the interval [0, maxlag[.
            Possible values are:

                * `'even'` (default) finds `n_lags` same width bins
                * `'uniform'` forms `n_lags` bins of same data count
                * `'fd'` applies Freedman-Diaconis estimator to find `n_lags`
                * `'sturges'` applies Sturge's rule to find `n_lags`.
                * `'scott'` applies Scott's rule to find `n_lags`
                * `'doane'` applies Doane's extension to Sturge's rule
                  to find `n_lags`
                * `'sqrt'` uses the square-root of
                  :func:`distance <skgstat.Variogram.distance>`
                  as `n_lags`.
                * `'kmeans'` uses KMeans clustering to well supported bins
                * `'ward'` uses hierarchical clustering to find
                  minimum-variance clusters.

            More details are given in the documentation for
            :func:`set_bin_func <skgstat.Variogram.set_bin_func>`.

        normalize : bool
            Defaults to False. If True, the independent and dependent
            variable will be normalized to the range [0,1].
        fit_method : str | None
            .. versionchanged:: 0.3.10
                Added 'ml' and 'custom'

            String identifying the method to be used for fitting the
            theoretical variogram function to the experimental. If None
            is passed, the fit does not run. More info is given in the
            Variogram.fit docs. Can be one of:

                * 'lm': Levenberg-Marquardt algorithm for unconstrained
                  problems. This is the faster algorithm, yet is the fitting of
                  a variogram not unconstrianed.
                * 'trf': Trust Region Reflective function for non-linear
                  constrained problems. The class will set the boundaries
                  itself. This is the default function.
                * 'ml': Maximum-Likelihood estimation. With the current
                  implementation only the Nelder-Mead solver for
                  unconstrained problems is implemented. This will estimate
                  the variogram parameters from a Gaussian parameter space
                  by minimizing the negative log-likelihood.
                * 'manual': Manual fitting. You can set the range, sill and
                  nugget either directly to the
                  :func:`fit <skgstat.Variogram.fit>` function, or as
                  `fit_` prefixed keyword arguments on Variogram instantiation.

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
            variogram function. For a sum of variogram, the nugget is defined
            in its last model.
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
        samples : float, int
            If set to a non-None value point pairs are sampled
            randomly. Two random subset of all points are chosen, and
            the distance matrix is calculated only between these two
            subsets. The size of each subset is set by `samples`: if <
            1 it specifies a fraction of all points, if >= 1 it
            specifies the number of points in each subset.
        n_lags : int
            Specify the number of lag classes to be defined by the binning
            function.
        verbose : bool
            Set the Verbosity of the class. Not Implemented yet.

        Keyword Arguments
        -----------------
        entropy_bins : int, str
            .. versionadded:: 0.3.7

            If the `estimator <skgstat.Variogram.estimator>` is set to
            `'entropy'` this argument sets the number of bins, that should be
            used for histogram calculation.
        percentile : int
            .. versionadded:: 0.3.7

            If the `estimator <skgstat.Variogram.estimator>` is set to
            `'entropy'` this argument sets the percentile to be used.
        binning_random_state : int, None
            .. versionadded:: 0.3.9

            If :func:`bin_func <skgstat.Variogram.set_bin_func>` is `'kmeans'`
            this can overwrite the seed for the initial guess of the cluster
            centroids. Note, that K-Means is not deterministic and is therefore
            seeded to 42 here. You can pass `None` to disable this behavior,
            but use it with care, as you will get different results.
        binning_agg_func : str
            .. versionadded:: 0.3.10

            If :func:`bin_func <skgstat.Variogram.set_bin_func>` is `'ward'`
            this keyword argument can switch from default mean aggregation to
            median aggregation for calculating the cluster centroids.
        obs_sigma : int, float
            .. versionadded:: 0.6.0

            If set, the Variogram will use this sigma as the standard deviation
            of the observations passed as values. Using a MonteCarlo simulation
            the uncertainties are propagated into the experimental variogram.
            If present, the plot will indicate the confidence interval as
            error bars around the experimental variogram.

        fit_bounds: 2-tuple of array_like or Bounds, optional
            .. versionadded:: 1.0.12

            Lower and upper bounds on parameters passed to scipy.optimize.curve_fit.

            Order is typically (range, sill, nugget) or (range, sill, smoothness, nugget) for individual models, or
            (range1, sill1, nugget1, range2, sill2, nugget2) for a sum of 2 models.
            Recommended for custom models, where bounds cannot be determined logically.
            For internal models, defaults to known min/max values for the sill (0, max variance), range (0, max lag)
            and smoothness (0, 2) or (0, 20) for stable and matern, respectively.

        fit_p0: array_like, optional
            .. versionadded:: 1.0.12

            Initial guess for the parameters passed to scipy.optimize.curve_fit.

            Same order as for fit_bounds.
            Defaults to upper bounds values. For custom models, if no bounds are defined, defaults to 1.
        """
        # Before we do anything else, make kwargs available
        self._kwargs = self._validate_kwargs(**kwargs)

        # handle the coordinates
        self._1d = False
        if not isinstance(coordinates, MetricSpace):
            coordinates = np.asarray(coordinates)

            # handle 1D coords
            if len(coordinates.shape) < 2:
                coordinates = np.column_stack((
                    coordinates,
                    np.zeros(len(coordinates))
                ))
                self._1d = True

            # handle maxlag for MetricSpace
            if maxlag and not isinstance(maxlag, str) and maxlag >= 1:
                _maxlag = maxlag
            else:
                _maxlag = None

            if samples is None:
                coordinates = MetricSpace(
                    coordinates.copy(),
                    dist_func,
                    _maxlag
                )
            else:
                coordinates = ProbabalisticMetricSpace(
                    coordinates.copy(),
                    dist_func, _maxlag,
                    samples=samples,
                    rnd=self._kwargs.get("binning_random_state", None)
                )
        elif dist_func != coordinates.dist_metric:
            raise AttributeError((
                "Distance metric of variogram differs "
                "from distance metric of coordinates"
            ))

        # Set coordinates
        self._X = coordinates

        # pairwise differences
        self._diff = None

        # set verbosity
        self.verbose = verbose

        # declare a flag to mark if this is a covariogram
        # this is set to None, as set_values will figure out.
        self._is_cross = None
        self._co_variable = None

        # set values
        self._values = None
        # calc_diff = False here, because it will be calculated by fit() later
        self.set_values(values=values, calc_diff=False)

        # lags and max lag
        self._n_lags_passed_value = n_lags
        self._n_lags = None
        self.n_lags = n_lags
        self._maxlag = None
        self.maxlag = maxlag

        # harmonize model placeholder
        self._harmonize = False

        # estimator can be a function or a string
        self._estimator = None
        self.set_estimator(estimator_name=estimator)

        # the binning settings
        self._bin_func_name = None
        self._bin_func = None
        self._groups = None
        self._bins = None
        self._bin_count = None
        self.set_bin_func(bin_func=bin_func)

        # Needed for harmonized models, no need to
        # run if there is no fit
        if fit_method is not None:
            self.preprocessing(force=True)

        # set if nugget effect shall be used
        self._use_nugget = None
        self.use_nugget = use_nugget

        # model can be a function or a string
        self._model = None
        self._model_name = None
        self._is_model_custom = False
        self.set_model(model_name=model)

        # specify if the lag should be given absolute or relative to the maxlag
        self._normalized = normalize

        # set the fitting method and sigma array
        self._fit_method = None
        if fit_method is not None:
            self.fit_method = fit_method
        self._fit_sigma = None
        self.fit_sigma = fit_sigma

        # set attributes to be filled during calculation
        self.cov = None
        self.cof = None

        # settings, not reachable by init (not yet)
        self._cache_experimental = False

        # do the preprocessing and fitting upon initialization
        # Note that fit() calls preprocessing
        fit_bounds = self._kwargs.get('fit_bounds') # returns None if empty
        fit_p0 = self._kwargs.get('fit_p0')
        self.fit(force=True, bounds=fit_bounds, p0=fit_p0)

        # finally check if any of the uncertainty propagation kwargs are set
        self._experimental_conf_interval = None
        self._model_conf_interval = None
        if 'obs_sigma' in self._kwargs:
            self._propagate_obs_sigma()

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
        return self._X.coords

    @property
    def metric_space(self):
        r"""
        .. versionadded:: 0.5.6

        :class:`MetricSpace <skgstat.MetricSpace>` representation of the
        input coordinates. A :class:`MetricSpace <skgstat.MetricSpace>`
        can be used to pass pre-calculated coordinates to other
        :class:`Variogram <skgstat.Variogram>` instances.

        Returns
        -------
        metric_space : skgstat.MetricSpace

        See Also
        --------
        Variogram.coordinates : coordinate representation

        """
        return self._X

    @metric_space.setter
    def metric_space(self, value):
        raise AttributeError('MetricSpace is read-only.')

    @property
    def dim(self):
        """
        Input coordinates dimensionality.
        """
        if self._1d:
            return 1
        return self.coordinates.shape[1]

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
        return squareform(self.pairwise_diffs)

    def set_values(self, values, calc_diff=True):
        """Set new values

        Will set the passed array as new value array. This array has to be of
        same length as the first axis of the coordinates array. The Variogram
        class does only accept one dimensional arrays.
        On success all fitting parameters are deleted and the pairwise
        differences are recalculated.
        Raises :py:class:`ValueError`s on shape mismatches and a Warning

        .. versionchanged::
            Now a warnings.warn message is thrown if all input data is the same

        Parameters
        ----------
        values : numpy.ndarray

        Returns
        -------
        void

        Raises
        ------
        ValueError : raised if the values array shape does not match the
            coordinates array, or more than one dimension given
        Warning : raised if all input values are the same

        See Also
        --------
        Variogram.values

        """
        # check dimensions
        if not len(values) == len(self.coordinates):  # pragma: no cover
            raise ValueError('The length of the values array has to match' +
                             'the length of coordinates')

        # use an array
        _y = np.asarray(values)

        # check if this is should be a cross-variogram
        # TODO: run tests for this
        if _y.ndim > 2:
            raise ValueError(
                "values has to be 1d (classic variogram) or 2d dimensional (cross-variogram)"
            )
        elif _y.ndim == 2:
            if _y.shape[1] > 2:
                raise ValueError(
                    "Use the utility function to create a grid of cross-variograms"
                )
            elif _y.shape[1] == 2:
                # set the co-variable
                self._co_variable = _y[:, 1].flatten()

                # generate a warning if the cross variogram flag was set to False
                if self._is_cross is not None and not self._is_cross:
                    warnings.warn("You passed two variables as observation and " +
                        "effectively turned this instance into a cross-variogram.")
                self._is_cross = True

                # by definition, the first axis is the observation
                _y = _y[:, 0].flatten()
        else:
            self._is_cross = False

        # check if all input values are the same
        if len(set(_y)) < 2:
            self.__single_input = True
            warnings.warn('All input values are the same.')
        else:
            self.__single_input = False

        # reset fitting parameter
        self.cof, self.cov = None, None
        self._diff = None

        # set new values
        self._values = np.asarray(_y)

        # recalculate the pairwise differences
        if calc_diff:
            self._calc_diff(force=True)

    @property
    def pairwise_diffs(self):
        """
        .. versionadded:: 1.0.4

        Pairwise residual differences of the input data.
        The property should be used over the Variogram._diff attribute,
        as this will contain multiple targets with future releases to
        implement cross-variograms.

        """
        if self._diff is None:
            self.preprocessing()

        return self._diff

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

    def set_bin_func(self, bin_func: Union[str, Iterable, Callable[[np.ndarray, float, float], Tuple[np.ndarray, float]]]):
        r"""Set binning function

        Sets a new binning function to be used. The new binning method is set
        by either a string identifying the new function to be used, or an
        iterable containing the bin edges, or any function that can compute
        bins from the distances, number of lags and maximum lag.
        The string can be one of: ['even', 'uniform', 'fd',
         'sturges', 'scott', 'sqrt', 'doane'].
        If the number of lag classes should be estimated automatically, it is
        recommended to use ' sturges' for small, normal distributed locations
        and 'fd' or 'scott' for large datasets, where 'fd' is more robust to
        outliers. 'sqrt' is by far the fastest estimator. 'doane' is an
        extension of Sturge's rule for non-normal distributed data.

        .. versionchanged:: 0.3.8
            added 'fd', 'sturges', 'scott', 'sqrt', 'doane'

        .. versionchanged:: 0.3.9
            added 'kmeans', 'ward'

        .. versionchanged:: 0.4.0
            added 'stable_entropy'

        .. versionchanged:: 0.4.1
            refactored local wrapper function definition. The wrapper to
            pass kwargs to the binning functions is now implemented as
            a instance method, to make it pickleable.

        .. versionchanged:: 0.6.5
            added iterable and function as arguments to allow for custom
            bins.

        Parameters
        ----------
        bin_func : str | Iterable | Callable
            Can be one of:

                * 'even'
                * 'uniform'
                * 'fd'
                * 'sturges'
                * 'scott'
                * 'sqrt'
                * 'doane'
                * 'kmeans'
                * 'ward'
                * 'stable_entropy'

        Returns
        -------
        void

        Notes
        -----
        **`'even'`**: Use skgstat.binning.even_width_lags for using
        n_lags lags of equal width up to maxlag.

        **`'uniform'`**: Use skgstat.binning.uniform_count_lags for using
        n_lags lags up to maxlag in which the pairwise differences
        follow a uniform distribution.

        **`'sturges'`**: estimates the number of evenly distributed lag
        classes (n) by Sturges rule [101]_:

        .. math::
            n = log_2 n + 1

        **`'scott'`**: estimates the lag class widths (h) by
        Scott's rule [102]_:

        .. math::
            h = \sigma \frac{24 * \sqrt{\pi}}{n}^{\frac{1}{3}}

        **`'sqrt'`**: estimates the number of lags (n) by the suare-root:

        .. math::
            n = \sqrt{n}

        **`'fd'`**: estimates the lag class widths (h) using the
        Freedman Diaconis estimator [103]_:

        .. math::
            h = 2\frac{IQR}{n^{1/3}}

        **`'doane'`**: estimates the number of evenly distributed lag classes
        using Doane's extension to Sturge's rule [104]_:

        .. math::
            n = 1 + \log_{2}(s) + \log_2\left(1 + \frac{|g|}{k}\right)
            g = E\left[\left(\frac{x - \mu_g}{\sigma}\right)^3\right]
            k = \sqrt{\frac{6(s - 2)}{(s + 1)(s + 3)}}

        **`'kmeans'`**: This method will search for `n` clusters in the
        distance matrix. The cluster centroids are used to calculate the
        upper edges of the lag classes, by setting it to half of the distance
        between two neighboring clusters. Note: This does not necessarily
        result in even width bins.

        **`'ward'`** uses a hierarchical culstering algorithm to iteratively
        merge pairs of clusters until there are only `n` remaining clusters.
        The merging is done by minimizing the variance for the merged cluster.

        **`'stable_entropy'`** will adjust `n` bin edges by minimizing the
        absolute differences between each lag's Shannon Entropy. This will
        lead to uneven bin widths. Each lag class value distribution will be
        of comparable intrinsic uncertainty from an information theoretic
        point of view, which makes the semi-variances quite comparable.
        However, it is not guaranteed, that the binning makes any sense
        from a geostatistical point of view, as the first lags might be way
        too wide.

        See Also
        --------
        Variogram.bin_func
        skgstat.binning.uniform_count_lags
        skgstat.binning.even_width_lags
        skgstat.binning.auto_derived_lags
        skgstat.binning.kmeans
        skgstat.binning.ward
        sklearn.cluster.KMeans
        sklearn.cluster.AgglomerativeClustering

        References
        ----------
        .. [101] Scott, D.W. (2009), Sturges' rule. WIREs Comp Stat, 1:
            303-306. https://doi.org/10.1002/wics.35
        .. [102] Scott, D.W. (2010), Scott's rule. WIREs Comp Stat, 2:
            497-502. https://doi.org/10.1002/wics.103
        .. [103] Freedman, David, and Persi Diaconis  (1981), "On the
            histogram as a density estimator: L 2 theory." Zeitschrift
            für Wahrscheinlichkeitstheorie und verwandte Gebiete 57.4:
            453-476.
        .. [104] Doane, D. P. (1976). Aesthetic frequency classifications.
            The American Statistician, 30(4), 181-183.

        """
        # handle strings
        if isinstance(bin_func, str):
            fname = bin_func.lower()
            # switch the input
            if fname == 'even':
                self._bin_func = binning.even_width_lags

            elif fname == 'uniform':
                self._bin_func = binning.uniform_count_lags

            # remove the n_lags if they will be adjusted on call
            else:
                # reset lags for adjusting algorithms
                if fname not in ('kmeans', 'ward', 'stable_entropy'):
                    self._n_lags = None

                # use the wrapper for all but even and uniform
                self._bin_func = self._bin_func_wrapper
            bin_func_name = bin_func

        elif isinstance(bin_func, Callable):  # pragma: no cover
            self._bin_func = bin_func
            bin_func_name = 'custom_func'

        elif isinstance(bin_func, Iterable):
            self._bin_func = None
            bin_func_name = 'custom_bin_edges'

        else:
            raise AttributeError('bin_func has to be of type string, iterable or callable.')

        # store the name
        self._bin_func_name = bin_func_name

        # reset groups, bins and bin count
        self._groups = None
        self._bin_count = None

        if isinstance(bin_func, str) or isinstance(bin_func, Callable):
            self._bins = None
        # if the input is an iterable with bin edges, we need to write self._bins here
        else:
            self._bins = bin_func
            self._maxlag = max(bin_func)
            self._n_lags = sum(1 for e in bin_func)

        self.cof, self.cov = None, None

    def _bin_func_wrapper(self, distances, n, maxlag):
        """
        Wrapper around the call of the actual binning method.
        This is needed to pass keyword arguments to kmeans or
        stable_entropy binning methods, and respect the slightly
        different function signature of auto_derived_lags.
        """
        if self._bin_func_name.lower() == 'kmeans':
            return binning.kmeans(distances, n, maxlag, **self._kwargs)

        elif self._bin_func_name.lower() == 'ward':
            return binning.ward(distances, n, maxlag)

        elif self._bin_func_name.lower() == 'stable_entropy':
            return binning.stable_entropy_lags(
                distances,
                n,
                maxlag,
                **self._kwargs
            )

        else:
            return binning.auto_derived_lags(
                distances,
                self._bin_func_name,
                maxlag
            )

    @property
    def normalized(self):
        return self._normalized

    @normalized.setter
    def normalized(self, status):
        # set the new value
        self._normalized = status

    @property
    def bins(self):
        """Distance lag bins

        Independent variable of the the experimental variogram sample.
        The bins are the upper edges of all calculated distance lag
        classes. If you need bin centers, use
        :func:`get_empirical <skgstat.Variogram.get_empirical>`.

        Returns
        -------
        bins : numpy.ndarray
            1D array of the distance lag classes.

        See Also
        --------
        Variogram.get_empirical

        """
        # if bins are not calculated, do it
        if self._bins is None:
            self._bins, n = self.bin_func(self.distance, self._n_lags, self.maxlag)
            # if the binning function returned an N, the n_lags need
            # to be adjusted directly (not through the setter)
            if n is not None:
                self._n_lags = n

        return self._bins.copy()

    @bins.setter
    def bins(self, bins):
        # set the new bins
        self._bins = np.asarray(bins)

        # clean the groups as they are not valid anymore
        self._bin_count = None
        self._groups = None
        self.cov = None
        self.cof = None

    @property
    def n_lags(self):
        """Number of lag bins

        Pass the number of lag bins to be used on
        this Variogram instance. This will reset
        the grouping index and fitting parameters

        """
        if self._n_lags is None:
            self._n_lags = len(self.bins)
        return self._n_lags

    @n_lags.setter
    def n_lags(self, n):
        # TODO: here accept strings and implement some optimum methods
        # string are not implemented yet
        if isinstance(n, str):  # pragma: no cover
            raise NotImplementedError('n_lags string values not implemented')

        # n_lags is int
        elif isinstance(n, int):
            if n < 1:
                raise ValueError('n_lags has to be a positive integer')

            # set parameter
            self._n_lags = n

            # reset the bins
            self._bins = None

        # else
        else:
            raise ValueError('n_lags has to be a positive integer')

        # if there are no errors, store the passed value
        self._n_lags_passed_value = n

        # reset the groups and bin count
        self._groups = None
        self._bin_count = None

        # reset the fitting
        self.cof = None
        self.cov = None

    @property
    def bin_count(self):

        if self._bin_count is None:
            self._bin_count = np.fromiter((g.size for g in self.lag_classes()), dtype=int)
        return self._bin_count

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
                    (
                        'Variogram estimator %s is not understood, please '
                        'provide the function.'
                    ) % estimator_name
                )
        elif callable(estimator_name):  # pragma: no cover
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
            # at first reset harmonize
            self._harmonize = False
            if model_name.lower() == 'harmonize':
                mname = 'harmonize'
                self._harmonize = True
                self._model = self._build_harmonized_model()
            elif "+" in model_name:
                mname =  ''.join(model_name.split()).lower() # Remove all whitespaces
                self._model = self._build_sum_models(mname)
            elif hasattr(models, model_name.lower()):
                mname = model_name.lower()
                self._model = getattr(models, mname)
            else:
                raise ValueError(
                    (
                        'The theoretical Variogram function %s is not'
                        ' understood, please provide the function'
                    ) % model_name
                )
            # Set model name attribute
            self._model_name = mname

        else:  # pragma: no cover
            self._is_model_custom = True
            self._model = model_name
            self._model_name = model_name.__name__

    def _get_argpos_sum_models(self, list_model_names):
        """
        Get argument slice position (list of slices) for the sum of models from a list of model names (list of strings).
        """

        # Doing this here for other functions (fit, describe, etc), even though already done in _build_sum_models
        list_models = [getattr(models, model_name.lower()).py_func for model_name in list_model_names]

        # Get the number of arguments per model (e.g., [3, 4, 4])
        nb_args_per_model = np.array([len(inspect.getfullargspec(model).args) for model in list_models])

        # We remove the nugget and lags parameters, except nugget for the last model (sum of nuggets = single nugget)
        nb_args_per_model -= 2
        nb_args_per_model[-1] += 1
        # Compute cumulative number of args removing 2 args everywhere (all lags and nuggets, last one will compensate)
        cum_args_minus_lag = np.cumsum(nb_args_per_model)

        # Prepare argument slices to distribute to submodels
        args_indices = np.insert(cum_args_minus_lag, 0, 0)  # We add the first indice of 0
        args_slices = [(slice(args_indices[i], args_indices[i + 1])) for i in range(len(args_indices) - 1)]

        return args_slices

    def _build_sum_models(self, sum_models_name: str):
        """
        Build sum of theoretical models, variogram-decorated function.
        """

        # Remove all whitespaces in the string, in case the user wrote something like "spherical + gaussian"
        sum_models_name = ''.join(sum_models_name.split()).lower()

        # Get individual model names
        list_model_names = sum_models_name.split("+")

        # Check that all models exist in the "models" module
        if not all(hasattr(models, model_name.lower()) for model_name in list_model_names):
            raise ValueError(
                (
                    'One of the theoretical models in the list "%s" is not'
                    ' understood, please provide existing model names separated by "+".'
                ) % ", ".join(list_model_names)
            )

        # First, build the models from their py_func (ignoring variogram decorator) and get args per model
        list_models = [getattr(models, model_name.lower()).py_func for model_name in list_model_names]
        # Get the argument positions for the model sum (function uses model names to be called by other methods)
        args_slices = self._get_argpos_sum_models(list_model_names=list_model_names)

        # Distribute first argument (lag) and use all others in order (nugget ignored when last argument not passed)
        @models.variogram
        def sum_models(h, *args):
            return sum(list_models[i](h, *args[args_slices[i]]) for i in range(len(list_models)))

        return sum_models

    def _build_harmonized_model(self):
        x = self.bins
        y = self.experimental

        _x = x[~np.isnan(y)]
        _y = y[~np.isnan(y)]
        regr = IsotonicRegression(increasing=True).fit(_x, _y)

        # create the model function
        def harmonize(x):
            """Monotonized Variogram

            Return the isotonic harmonized experimental variogram.
            This means, the experimental variogram is monotonic
            after harmonization.

            The harmonization is done using following Hinterding (2003)
            using the PAVA algorithm (Barlow and Bartholomew, 1972).

            Returns
            -------
            gamma : numpy.ndarray
                monotonized experimental variogram

            References
            ----------
            Barlow, R., D. Bartholomew, et al. (1972): Statistical
                Interference Under Order Restrictions. John Wiley
                and Sons, New York.
            Hiterding, A. (2003): Entwicklung hybrider Interpolations-
                verfahren für den automatisierten Betrieb am Beispiel
                meteorologischer Größen. Dissertation, Institut für
                Geoinformatik, Westphälische Wilhelms-Universität Münster,
                IfGIprints, Münster. ISBN: 3-936616-12-4

            """

            if isinstance(x, (list, tuple, np.ndarray)):
                return regr.transform(x)
            else:
                return regr.transform([x])

        return harmonize

    @property
    def use_nugget(self):
        """
        Use a nugget effect on this Variogram instance.
        If disabled, the automatic fitting procedures will omit the nugget and
        not use it as a model parameter.

        Note
        ----
        If :func:`fit_method <skgstat.Variogram.fit_method>` is set to
        ``'manual'`` and a nugget parameter is pass to
        :func:`fit <skgstat.Variogram.fit>`, use_nugget will be set to True.

        Returns
        -------
        use_nugget : bool

        """
        return self._use_nugget

    @use_nugget.setter
    def use_nugget(self, enable_nugget):
        if not isinstance(enable_nugget, bool):
            raise ValueError('use_nugget has to be of type bool.')

        # set new value
        self._use_nugget = enable_nugget

    @property
    def dist_function(self):
        return self._X.dist_metric

    @classmethod
    def wrapped_distance_function(cls, dist_func, x, **kwargs):
        return pdist(X=x, metric=dist_func, **kwargs)

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
        # reset the fitting
        self.cof, self.cov = None, None

        if isinstance(func, str):  # pragma: no cover
            if func.lower() == 'rank':
                raise NotImplementedError
        elif not callable(func):
            raise ValueError('Input not supported. Pass a string or callable.')

        # re-calculate distances
        self._X = MetricSpace(self._X.coords, func, self._X.max_dist)

    @property
    def distance(self):
        # handle sparse matrix
        if isinstance(self.distance_matrix, sparse.spmatrix):
            return self.triangular_distance_matrix.data

        # Turn it back to triangular form not to have duplicates
        return squareform(self.distance_matrix)

    @property
    def triangular_distance_matrix(self):
        """
        Like distance_matrix but with zeros below the diagonal...
        Only defined if distance_matrix is a sparse matrix
        """
        if not isinstance(self.distance_matrix, sparse.spmatrix):
            raise RuntimeWarning("Only available for sparse coordinates.")

        m = self.distance_matrix
        c = m.tocsc()
        c.data = c.indices
        rows = c.tocsr()
        filt = sparse.csr_matrix(
            (m.indices < rows.data, m.indices, m.indptr),
            m.shape
        )
        return m.multiply(filt)

    @property
    def distance_matrix(self):
        return self._X.dists

    @property
    def maxlag(self):
        """
        Maximum lag distance to be considered in this Variogram instance.
        You can limit the distance at which point pairs are calculated.
        There are three possible ways how to do that, in absolute lag units,
        which is a number larger one. Secondly, a number ``0 < maxlag < 1``
        can be set, which will use this share of the maximum distance as
        maxlag. Lastly, a string can be set: ``'mean'`` and ``'median'``
        for the mean or median value of the distance matrix.

        Notes
        -----
        This setting is largely flexible, but all options except the
        absolute limit in lag units need the full distance matrix to be
        calculated. Hence, it does **not** speed up the calculation
        of large distance matrices, just the estimation of the variogram.
        Thus, if you pre-calcualte the distance matrix using
        :class:`MetricSpace <skgstat.MetricSpace>`, only absolute
        limits can be used.

        """
        return self._maxlag

    @maxlag.setter
    def maxlag(self, value):
        # reset fitting
        self.cof, self.cov = None, None

        # remove bins, groups, and bin count
        self._bins = None
        self._groups = None
        self._bin_count = None

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
    def fit_method(self):
        """
        .. versionadded:: 0.6.2

        Set the fit method to be used for this Variogram instance.
        Possible values are:

        * ``'trf'``   - Trust-Region Reflective (default)
        * ``'lm'``    - Levenberg-Marquardt
        * ``'ml'``    - Maximum Likelihood estimation
        * `'manual'`` - Manual fitting by setting the parameters

        .. versionchanged:: 0.6.6
            Passing `None` will prevent the fitting procedure from
            running.

        See Also
        --------
        scipy.optimize.minimize
        scipy.optimize.curve_fit
        Variogram.fit

        Notes
        -----
        The default method (TRF) is a bounded least squares method,
        that sets constraints to the value space of all parameters.
        All methods use an initial guess for all used parameters.
        This is :func:`max(bins) <skgstat.Variogram.bins>` for
        the range,
        :func:`max(experimental) <skgstat.Variogram.experimental>`
        for the sill, ``20`` for the Matérn smoothness, ``2`` for
        the stable model shape and ``1`` for the nugget if used.

        """
        return self._fit_method

    @fit_method.setter
    def fit_method(self, value):
        if value not in ('lm', 'ml', 'trf', 'manual'):
            raise AttributeError(
                "fit_method has to be one of ['lm', 'ml', 'trf', 'manual']"
            )

        # value is fine -check for manual does not drop the coefficients
        elif value == 'manual':
            self._fit_method = 'manual'

        # Trust region reflective does not work wi
        elif value == 'trf' and self.__single_input:
            raise AttributeError("'trf' is bounded and therefore not supported when all values are the same.")

        # otherwise - refit
        else:
            # set method
            self._fit_method = value

            # reset fit
            self.cof = None
            self.cov = None

    @property
    def fit_sigma(self):
        r"""Fitting Uncertainty

        Set or calculate an array of observation uncertainties aligned to the
        Variogram.bins. These will be used to weight the observations in the
        cost function, which divides the residuals by their uncertainty.

        When setting fit_sigma, the array of uncertainties itself can be
        given, or one of the strings:
        ['linear', 'exp', 'sqrt', 'sq', 'entropy'].
        The parameters described below refer to the setter of this property.

        .. versionchanged:: 0.3.11
            added the 'entropy' option.

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
              * **sigma='entropy'**: Calculates the Shannon Entropy as
                intrinsic uncertainty of each lag class.

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
                raise AttributeError('len(fit_sigma) != len(bins)')

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

        # entropy
        elif self._fit_sigma == 'entropy':
            # get the binning using scotts rule
            bins = np.histogram_bin_edges(self.distance, 'scott')

            # get the maximum entropy
#            hmax = np.log2(len(self.distance))

            # apply the entropy
            h = np.asarray([shannon_entropy(grp, bins) for grp in self.lag_classes() if len(grp) > 0])
            return 1. / h

        else:
            raise ValueError(
                "fit_sigma is not understood. It has to be an " +
                "array or one of ['linear', 'exp', 'sqrt', 'sq', 'entropy']."
            )

    @fit_sigma.setter
    def fit_sigma(self, sigma):
        self._fit_sigma = sigma

        # remove fitting parameters
        self.cof = None
        self.cov = None

    @property
    def is_cross_variogram(self) -> bool:
        """Read-only flag indicating if the current instance is a cross-variogram"""
        return self._is_cross

    def update_kwargs(self, **kwargs):
        """
        .. versionadded:: 0.3.7

        Update the keyword arguments of this Variogram instance.
        The keyword arguments will be validated first and the update the
        existing kwargs. That means, you can pass only the kwargs, which
        need to be updated.

        .. note::
            Updating the kwargs does not force a preprocessing circle.
            Any affected intermediate result, that might be cached internally,
            will not make use of updated kwargs. Make a call to
            :func:`preprocessing(force=True) <skgstat.Variogram.preprocessing>`
            to force a clean re-calculation of the Variogram instance.

        """
        old = self._kwargs

        # update the keyword-arguments
        updated = self._validate_kwargs(**kwargs)
        old.update(updated)

        self._kwargs = old

    def _validate_kwargs(self, **kwargs):
        """
        .. versionadded:: 0.3.7

        This functions actually does nothing right now.
        It will be used in the future, as soon as the Variogram takes
        more kwargs. Then, these can be checked here.

        """
        return kwargs

    def lag_groups(self):
        """Lag class groups

        Returns a mask array with as many elements as self._diff has,
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

        .. versionchanged:: 0.3.6
            yields an empty array for empty lag groups now

        Returns
        -------
        iterable

        """
        # get the diffs
        diffs = self.pairwise_diffs

        # get the groups
        groups = self.lag_groups()

        # yield all groups 
        for i in range(len(self.bins)):
            yield diffs[np.where(groups == i)]

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
        self._calc_diff(force=force)
        self._calc_groups(force=force)

    def fit(self, force=False, method=None, sigma=None, bounds=None, p0=None, **kwargs):
        """Fit the variogram

        The fit function will fit the theoretical variogram function to the
        experimental. The preprocessed distance matrix, pairwise differences
        and binning will not be recalculated, if already done. This could be
        forced by setting the force parameter to true.

        In case you call fit function directly, with method or sigma,
        the parameters set on Variogram object instantiation will get
        overwritten. All other keyword arguments will be passed to
        scipy.optimize.curve_fit function.

        .. versionchanged:: 0.3.10
            added 'ml' and 'custom' method.

        .. versionchanged:: 1.0.1
            use_nugget is now flagged implicitly, whenever a nugget > 0 is
            passed in manual fitting.

        Parameters
        ----------
        force : bool
            If set to True, a clean preprocessing of the distance matrix,
            pairwise differences and the binning will be forced. Default is
            False.
        method : string
            A string identifying one of the implemented fitting procedures.
            Can be one of:

              * lm: Levenberg-Marquardt algorithms implemented in
                scipy.optimize.leastsq function.
              * trf: Trust Region Reflective algorithm implemented in
                scipy.optimize.least_squares(method='trf')
              * 'ml': Maximum-Likelihood estimation. With the current
                implementation only the Nelder-Mead solver for unconstrained
                problems is implemented. This will estimate the variogram
                parameters from a Gaussian parameter space by minimizing
                the negative log-likelihood.
              * 'manual': Manual fitting. You can set the range, sill and
                nugget either directly to the
                :func:`fit <skgstat.Variogram.fit>` function, or as
                `fit_` prefixed keyword arguments on Variogram instantiation.


        sigma : string, array
            Uncertainty array for the bins. Has to have the same dimension as
            self.bins. Refer to Variogram.fit_sigma for more information.

        bounds: 2-tuple of array_like or Bounds, optional
            Lower and upper bounds on parameters passed to scipy.optimize.curve_fit.

            Order is typically (range, sill, nugget) or (range, sill, smoothness, nugget) for individual models, or
            (range1, sill1, nugget1, range2, sill2, nugget2) for a sum of 2 models.
            Recommended for custom models, where bounds cannot be determined logically.
            For internal models, defaults to known min/max values for the sill (0, max variance), range (0, max lag)
            and smoothness (0, 2) or (0, 20) for stable and matern, respectively.

        p0: array_like, optional
            Initial guess for the parameters passed to scipy.optimize.curve_fit.

            Same order as for fit_bounds.
            Defaults to upper bounds values. For custom models, if no bounds are defined, defaults to 1.

        Returns
        -------
        void

        See Also
        --------
        scipy.optimize.minimize
        scipy.optimize.curve_fit
        scipy.optimize.leastsq
        scipy.optimize.least_squares

        """
        # store the old cof
        if self.cof is None:
            old_params = {}
        else:
            old_params = self.describe()

        # delete the last cov and cof
        if force:
            self.cof = None
            self.cov = None

        # if force, force a clean preprocessing
        self.preprocessing(force=force)

        # load the data
        x = np.array(self.bins)
        y = self.experimental

        # overwrite fit setting if new params are given
        if method is not None:
            self.fit_method = method
        if sigma is not None:
            self.fit_sigma = sigma

        # remove nans
        _x = x[~np.isnan(y)]
        _y = y[~np.isnan(y)]

        # check if method is manual and a nugget was passed
        if self.fit_method == 'manual' and kwargs.get('nugget', False):
            self.use_nugget = True

        # handle harmonized models
        if self._harmonize:
            _x = np.linspace(0, np.max(_x), 100)
            _y = self._model(_x)

            # get the params
            s = 0.95 * np.nanmax(_y)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = np.argwhere(_y >= s)[0][0]
            n = _y[0] if self.use_nugget else 0.0

            # set the params
            self.cof = [r, s, n]
            return

        # For a supported model, wrap the function depending on nugget and get logical bounds
        if not self._is_model_custom:
            # Switch the method
            # wrap the model to include or exclude the nugget
            if self.use_nugget:
                def wrapped(*args):
                    return self._model(*args)
            else:
                def wrapped(*args):
                    return self._model(*args, 0)

            # get p0
            if bounds is None:
                bounds = (0, self.__get_fit_bounds(x, y))
            if p0 is None:
                p0 = np.asarray(bounds[1])
        # Else, inspect the function for the number of arguments
        else:
            # The number of arguments of argspec minus one is what we initialized
            argspec = inspect.getfullargspec(self._model.__wrapped__)
            nb_args = len(argspec.args) - 1
            if bounds is None:
                warnings.warn("Parameter bounds cannot be logically derived for a custom model during the fit. "
                              "User bounds can be passed using fit(..., bounds=) or Variogram(..., fit_bounds=).", UserWarning)
                bounds = ([-np.inf] * nb_args, [np.inf] * nb_args)
            if p0 is None:
                # If all bounds are infinite (not defined, pass 1)
                if bounds == ([-np.inf] * nb_args, [np.inf] * nb_args):
                    p0 = np.ones(nb_args)
                # Else pass the upper bounds
                else:
                    p0 = np.asarray(bounds[1])

            def wrapped(*args):
                return self._model(*args)

        # Trust Region Reflective
        if self.fit_method == 'trf':
            self.cof, self.cov = curve_fit(
                wrapped,
                _x, _y,
                method='trf',
                sigma=self.fit_sigma,
                p0=p0,
                bounds=bounds,
                **kwargs
            )

        # Levenberg-Marquardt
        elif self.fit_method == 'lm':
            self.cof, self.cov = curve_fit(
                wrapped,
                _x, _y,
                method='lm',
                sigma=self.fit_sigma,
                p0=p0,
                **kwargs
            )

        # maximum-likelihood
        elif self.fit_method == 'ml':
            # check if the probabilities must be weighted
            if self.fit_sigma is None:
                sigma = np.ones(self.bins.size)
            else:
                sigma = 1 / self.fit_sigma

            # define the loss function to be minimized
            def ml(params):
                # predict
                pred = [wrapped(_, *params) for _ in _x]

                # get the probabilities of _y
                p = [stats.norm.logpdf(_p, loc=o, scale=1.) for _p, o in zip(pred, _y)]

                # weight the probs
                return - np.sum(p * sigma)

            # apply maximum likelihood estimation by minimizing ml
            result = minimize(ml, p0 * 0.5, method='SLSQP', bounds=[(0, _) for _ in p0])

            if not result.success:  # pragma: no cover
                raise OptimizeWarning('Maximum Likelihood could not estimate parameters.')
            else:
                # set the result
                self.cof = result.x

        # manual fitting
        elif self.fit_method == 'manual':
            # TODO: here, the Error could only be raises if cof was None so far
            r = kwargs.get('range', self._kwargs.get('fit_range', old_params.get('effective_range')))
            s = kwargs.get('sill', self._kwargs.get('fit_sill', old_params.get('sill')))

            # if not given raise an AttributeError
            if r is None or s is None:
                raise AttributeError('For manual fitting, you need to pass the \
                    variogram parameters either to fit or to the Variogram \
                    instance.\n parameter need to be prefixed with fit_ if \
                    passed to __init__.')

            # get the nugget
            n = kwargs.get('nugget', self._kwargs.get('fit_nugget', old_params.get('nugget', 0.0)))

            # check if a s parameter is needed
            if self._model_name in ('stable', 'matern'):
                if self._model_name == 'stable':
                    s2 = kwargs.get('shape', self._kwargs.get('fit_shape', old_params.get('shape',2.0)))
                if self._model_name == 'matern':
                    s2 = kwargs.get('shape', self._kwargs.get('fit_shape', old_params.get('smoothness', 2.0)))

                # set
                self.cof = [r, s, s2, n]
            else:
                self.cof = [r, s, n]

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

        # return the result
        return self.fitted_model(x)

    @property
    def fitted_model(self):
        """Fitted Model

        Returns a callable that takes a distance value and returns a
        semivariance. This model is fitted to the current Variogram
        parameters. The function will be interpreted at return time with the
        parameters hard-coded into the function code.

        Returns
        -------
        model : callable
            The current semivariance model fitted to the current Variogram
            model parameters.
        """
        if self.cof is None:
            self.fit(force=True)

        return self.fitted_model_function(self._model, self.cof)

    @classmethod
    def fitted_model_function(cls, model, cof=None, **kw):
        if cof is None:
            # Make sure to keep this synchronized with the output
            # of describe()!
            cof = [kw["effective_range"], kw["sill"]]
            if "smoothness" in kw:
                cof.append(kw["smoothness"])
            if "shape" in kw:
                cof.append(kw["shape"])
            if kw["nugget"] != 0:
                cof.append(kw["nugget"])

        if not callable(model):
            if model == "harmonize":
                raise ValueError("Please supply the actual harmonized model directly")
            else:
                model = model.lower()
                model = getattr(models, model)

        if model.__name__ == "harmonize":
            code = """fitted_model = lambda x: model(x)"""
        else:
            code = """fitted_model = lambda x: model(x, %s)""" % \
               (', '.join([str(_) for _ in cof]))

        # run the code
        loc = dict(model=model)
        exec(code, loc, loc)
        return loc['fitted_model']

    def _format_values_stack(self, values: np.ndarray) -> np.ndarray:
        """
        Create a numpy column stack to calculate differences between two value arrays.
        The format function will handle sparse matrices, as these do not include
        pairwise differences that are separated beyond maxlag.
        The dense numpy.array matrices contain all point pairs.

        """
        # handle sparse matrix
        if isinstance(self.distance_matrix, sparse.spmatrix):
            # get triangular distance matrices
            c = r = self.triangular_distance_matrix

            # get the sparse CSR matrix
            if not isinstance(c, sparse.csr_matrix):
                c = c.tocsr()
            if not isinstance(r, sparse.csc_matrix):
                r = r.tocsc()

            # get the cols
            Vcol = sparse.csr_matrix(
                (values[c.indices], c.indices, c.indptr)
            )

            # The sparse rows are always the observations
            Vrow = sparse.csc_matrix(
                (values[r.indices], r.indices, r.indptr)
            ).tocsr()

            # self._diff will have same shape as self.distances, even
            # when that's not in diagonal format...
            # Note: it might be compelling to do np.abs(Vrow -
            # Vcol).data instead here, but that might optimize away
            # some zeros, leaving _diff in a different shape
            return np.abs(Vrow.data - Vcol.data)

        else:
            # Append a column of zeros to make pdist happy
            # euclidean: sqrt((a-b)**2 + (0-0)**2) == sqrt((a-b)**2)
            return pdist(
                np.column_stack((values, np.zeros(len(values)))),
                metric="euclidean"
            )

    def _calc_diff(self, force=False):
        """
        Calculates the pairwise differences for all coordinate locations.
        If the Variogram is a cross-variogram, the differences are calculated
        between the main variable (self.values) and the co-variable.

        .. versionchanged:: 1.0.5
            calculates co-variate differences

        Returns
        -------
        void

        """
        # already calculated
        if self._diff is not None and not force:
            return

        # format into column-stack for faster calculation
        diffs = self._format_values_stack(self.values)

        # check if this is a cross-variogram
        if self.is_cross_variogram:
            co_diffs = self._format_values_stack(self._co_variable)

            # multiply to cross-difference
            diffs *= co_diffs

        # set the new differences
        self._diff = diffs

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
        
        # -1 is the group for distances outside maxlag
        bins = np.digitize(d, bin_edges)
        self._groups = np.where(bins == self.n_lags, -1, bins) 

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
        return self._experimental

    @property
    def _experimental(self):
        """
        Calculates the experimental variogram from the current lag classes.
        It handles the special case of the `'entropy'` and `'percentile'`
        estimators, which take an additional argument.

        .. versionchanged:: 0.3.6
            replaced the for-loops with :func:`fromiter <numpy.fromiter>`

        .. versionchanged:: 0.3.7
            makes use of `kwargs <skgstat.Variogram._kwargs>` for
            specific estimators now

        Returns
        -------
        experimental : np.ndarray
            1D array of the experimental variogram values. Has same length
            as :func:`bins <skgstat.Variogram.bins>`

        """
        if self._estimator.__name__ == 'entropy':
            # get the parameter from kwargs, if not set use 50
            N = self._kwargs.get('entropy_bins', 50)

            # we need to use N -1 as we use the last inclusive
            if isinstance(N, int):
                N -= 1

            bins = np.histogram_bin_edges(self.distance, bins=N)

            # define the mapper to the estimator function
            def mapper(lag_values):
                return self._estimator(lag_values, bins=bins)

        elif self._estimator.__name__ == 'percentile':
            if self._kwargs.get('percentile', False):
                p = self._kwargs.get('percentile')

                def mapper(lag_values):
                    return self._estimator(lag_values, p=p)
            else:
                mapper = self._estimator

        else:
            mapper = self._estimator

        # return the mapped result
        return np.fromiter(map(mapper, self.lag_classes()), dtype=float)

    def get_empirical(self, bin_center=False):
        """Empirical variogram

        Returns a tuple of dependent and independent sample values, this
        :class:`Variogram <skgstat.Variogram>` is estimated for.
        This is a tuple of the current :func:`bins <skgstat.Variogram.bins>`
        and :func:`experimental <skgstat.Variogram.experimental>`
        semi-variance values. By default the upper bin edges are used.
        This can be set to bin center by the `bin_center` argument.

        Parameters
        ----------
        bin_center : bool
            If set to `True`, the center for each distance lag bin is
            used over the upper limit (default).

        Returns
        -------
        bins : numpy.ndarray
            1D array of :func:`n_lags <skgstat.Variogram.n_lags>`
            distance lag bins.
        experimental : numpy.ndarray
            1D array of :func:`n_lags <skgstat.Variogram.n_lags>`
            experimental semi-variance values.

        See Also
        --------
        Variogram.bins
        Variogram.experimental

        """
        # get the bins and experimental values
        bins = self.bins
        experimental = self.experimental

        # align bin centers
        if bin_center:
            # get the bin centers
            bins = np.subtract(bins, np.diff([0] + bins.tolist()) / 2)

        # return
        return bins, experimental

    def __get_fit_bounds(self, x, y):
        """
        Return the bounds for parameter space in fitting a variogram model.
        The bounds are depended on the Model that is used

        Returns
        -------
        list

        """
        all_mname = self._model_name

        # for a sum of models, create a list
        if "+" in all_mname:
            list_mname = all_mname.split("+")
        else:
            list_mname = [all_mname]

        # we append all bounds (for one or several models)
        all_bounds = []
        for i, mname in enumerate(list_mname):

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

            # if use_nugget is True add the nugget (for the last model only in case it is a sum)
            if self.use_nugget and i == (len(list_mname) - 1):
                bounds.append(0.99*np.nanmax(y))

            all_bounds += bounds

        return all_bounds

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
        _bin = self.bins

        x = np.linspace(0, np.float64(np.nanmax(_bin)), n)

        # fit if needed
        if self.cof is None:
            self.fit(force=force)

        return x, self._model(x, *self.cof)

    def _propagate_obs_sigma(self):
        """
        Observation uncertainties are propagated into the experimental
        variogram.
        """
        # due to circular import, import here
        from skgstat.util.uncertainty import propagate

        # TODO: load and propagate percentile settings
        self._experimental_conf_interval = propagate(
            self,
            'values',
            self._kwargs['obs_sigma'],
            use_bounds=True
        )

    @property
    def residuals(self):
        """Model residuals

        Calculate the model residuals defined as the differences between the
        experimental variogram and the theoretical model values at
        corresponding lag values

        .. deprecated:: 1.0.4
            residuals can be ambiguous, thus the property is renamed to model_residuals

        Returns
        -------
        numpy.ndarray

        """
        warnings.warn(
            "residuals is deprecated and will be removed. Please use Variogram.model_residuals",
            DeprecationWarning
        )
        return self.model_residuals


    @property
    def model_residuals(self) -> np.ndarray:
        """
        Calculate the model residuals defined as the differences between the
        experimental variogram and the theoretical model values at
        corresponding lag values.

        Returns
        -------
        residuals : numpy.ndarray

        """
        # get the deviations
        experimental, model = self.model_deviations()

        # calculate the residuals
        return np.fromiter(
            map(lambda x, y: x - y, model, experimental), float
        )

    @property
    def mean_residual(self):
        """Mean Model residuals

        Calculates the mean, absolute deviations between the experimental
        variogram and theretical model values.

        Returns
        -------
        float
        """
        return np.nanmean(np.fromiter(map(np.abs, self.model_residuals), float))

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
        return self.root_mean_square

    @property
    def mse(self):
        r"""MSE

        Calculate the Mean squared error between the experimental
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
        The MSE is implemented like:

        .. math::
            MSE = \frac{\sum_{i=0}^{i=N(x)} (x-y)^2}{N(x)}

        """
        # get the deviations
        experimental, model = self.model_deviations()

        # get the mean
        mse = np.mean(np.power(np.subtract(experimental, model), 2))

        return mse

    @property
    def mae(self):
        r"""RMSE

        Calculate the Mean absolute error between the experimental
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
        The MAE is implemented like:

        .. math::
            MAE = \frac{\sum_{i=0}^{i=N(x)} |x-y|}{N(x)}

        """
        # get the deviations
        experimental, model = self.model_deviations()

        # get the mean
        mae = np.mean(np.abs(np.subtract(experimental, model)))

        return mae

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
    def root_mean_square(self):
        """Root Mean Square (RMS) of the residuals

        Calculates the square root of the mean of squared residuals.

        Returns
        -------
        float
            Root Mean Square of the residuals.
        """
        return np.sqrt(np.nanmean(np.square(self.model_residuals)))

    @property
    def residual_sum_of_squares(self):
        """Residual Sum of Squares (RSS)

        Calculates the sum of squared differences between the experimental
        variogram and theoretical model values.

        Returns
        -------
        float
            Residual sum of squares (RSS), a measure of the overall model fit
            representing the sum of squared deviations between the observed
            experimental variogram and the corresponding theoretical model values.
        """
        return np.nansum(np.square(self.model_residuals))

    @property
    def rss(self):
        return self.residual_sum_of_squares

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
        but the difference of the maximum y to its mean:

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
        # get the experimental and theoretical variogram and calculate means
        experimental, model = self.model_deviations()
        mx = np.nanmean(experimental)
        my = np.nanmean(model)

        # calculate the single pearson correlation terms
        term1 = np.nansum(np.fromiter(
            map(lambda x, y: (x-mx) * (y-my), experimental, model), float
        ))

        t2x = np.nansum(
            np.fromiter(map(lambda x: (x-mx)**2, experimental), float)
        )
        t2y = np.nansum(
            np.fromiter(map(lambda y: (y-my)**2, model), float)
        )

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
        term1 = np.nansum(np.fromiter(
            map(lambda x, y: (x - y)**2, experimental, model),
            float
            )
        )
        term2 = np.nansum(np.fromiter(
            map(lambda x: (x - mx)**2, experimental),
            float
            )
        )

        return 1 - (term1 / term2)

    @property
    def aic(self) -> float:
        """
        """
        from skgstat.util.cross_validation import aic
        return aic(self)

    @property
    def bic(self) -> float:
        """
        """
        from skgstat.util.cross_validation import bic
        return bic(self)

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

    def cross_validate(
        self,
        method: str = 'jacknife',
        n: int = None,
        metric: str = 'rmse',
        seed=None
    ) -> float:
        """
        Cross validation of the variogram model by means of Kriging.
        Right now, this function can only utilize a jacknife (leave-one-out)
        cross validation and will only use the builtin OrdinaryKriging
        method (not yet the to_gs_krige interface).

        Parameters
        ----------
        method : str
            Right now, 'jacknife' is the only possible input.
        n : int
            The number of points to be included into the cross-validation.
            If None (default), all points will be used.
        metric : str
            Metric used for cross-validation.
            Can be root mean square error (rmse), mean squared error (mse)
            or mean absolute error (mae).
        seed : int
            If n is not None, the random selection of input data for the
            cross-validation can be seeded.

        Returns
        -------
        metric : float
            The cross-validation result as specified above.

        See Also
        --------
        skgstat.util.cross_validation.jacknife
        """
        # not sure how to solve the circular import here
        from skgstat.util import cross_validation

        if method == 'jacknife':
            return cross_validation.jacknife(
                self,
                n=n,
                metric=metric,
                seed=seed
            )
        else:
            raise AttributeError(f"A method '{method}' is not implemented.")

    def describe(self, short=False, flat=False):
        """Variogram parameters

        Return a dictionary of the variogram parameters.

        .. versionchanged:: 0.3.7
            The describe now returns all init parameters in as the
            `describe()['params']` key and all keyword arguments as
            `describe()['kwargs']`. This output can be suppressed
            by setting `short=True`.

        Parameters
        ----------
        short : bool
            If `True`, the `'params'` and `'kwargs'` keys will be
            omitted. Defaults to `False`.
        flat : bool
            If `True`, the `'params'` and `'kwargs'` nested `dict`s
            will be distributed to the main `dict` to return a
            flat `dict`. Defaults to `False`

        Returns
        -------
        parameters : dict
            Returns fitting parameters of the theoretical variogram
            model along with the init parameters of the
            `Variogram <skgstat.Variogram>` instance.

        """
        # fit, if not already done
        if self.cof is None:
            self.fit(force=True)

        # scale sill and range
        maxlag = np.nanmax(self.bins)
        maxvar = np.nanmax(self.experimental)

        # get the fitting coefficients
        cof = self.cof

        # build the dict

        def fnname(fn):
            if callable(fn):
                name = "%s.%s" % (fn.__module__, fn.__name__)
            else:
                name = fn
            if name.startswith("skgstat."):
                return name.split(".")[-1]
            return name

        rdict = dict(
            model=fnname(self._model) if not self._harmonize else "harmonize",
            estimator=fnname(self._estimator),
            dist_func=fnname(self.dist_function))

        def create_dict_for_model(model_name, cof, maxlag, maxvar, use_nugget, id = ''):
            """
            Create dictionary of parameters for an individual model.
            Optionally, append a model ID to the key (for the case of summed models).
            """
            # id appended to the param key name to differentiate models for a sum
            if id != '':
                id = '_' + id

            d = {
                'normalized_effective_range' + id: cof[0] * maxlag,
                'normalized_sill' + id: cof[1] * maxvar,
                'normalized_nugget' + id: cof[-1] * maxvar if use_nugget else 0,
                'effective_range' + id: cof[0],
                'sill' + id: cof[1],
                'nugget' + id: cof[-1] if use_nugget else 0,
            }

            # handle s parameters for matern and stable model
            if model_name == 'matern':
                d['smoothness' + id] = cof[2]
            elif model_name == 'stable':
                d['shape' + id] = cof[2]

            return d

        # for a custom model: we list the optimized params for the function wrapped by the variogram decorator
        if self._is_model_custom:
            custom_arg_names = inspect.getfullargspec(self._model.__wrapped__).args
            all_params = {"param"+str(i+1)+"_"+custom_arg_names[i+1]: cof[i] for i in range(len(custom_arg_names) - 1)}

        # for a sum of models
        elif "+" in self._model_name:
            list_model_names = self._model_name.split("+")
            list_argslices = self._get_argpos_sum_models(list_model_names=list_model_names)
            all_params = {}
            # add the parameters for each model, with parameter suffix from 1 to the total number
            for i in range(len(list_model_names)):
                model_params = create_dict_for_model(model_name=list_model_names[i], cof=cof[list_argslices[i]],
                                                  maxlag=maxlag, maxvar=maxvar, use_nugget=self.use_nugget, id=str(i+1))
                all_params.update(model_params)

        # for a single model
        else:
            all_params = create_dict_for_model(model_name=self._model_name, cof=cof, maxlag=maxlag, maxvar=maxvar,
                                                use_nugget=self.use_nugget)
        # update dictionary
        rdict.update(all_params)

        # add other stuff if not short version requested
        if not short:
            kwargs = self._kwargs
            params = dict(
                estimator=self._estimator.__name__,
                model=self._model_name,
                dist_func=str(self.dist_function),
                bin_func=self._bin_func_name,
                normalize=self.normalized,
                fit_method=self._fit_method,
                fit_sigma=self.fit_sigma,
                use_nugget=self.use_nugget,
                maxlag=self.maxlag,
                n_lags=self._n_lags_passed_value,
                verbose=self.verbose
            )

            # update or append the params
            if flat:
                rdict.update(params)
                rdict.update(kwargs)
            else:
                rdict['params'] = params
                rdict['kwargs'] = kwargs

        # return
        return rdict

    @property
    def parameters(self):
        """
        Extract just the variogram parameters range, sill
        and nugget from the
        :func:`describe <skgstat.Variogram.describe>` output.

        Returns
        -------
        params : list
            [range, sill, nugget] for most models and
            [range, sill, shape, nugget] for matern and stable model.
            [range1, sill1, nugget1, range2, still2, nugget2] for a sum of 2 models.
            [param1, param2, param3, ...] in order for a custom model.

        """
        d = self.describe()
        if 'error' in d:
            return [None, None, None]

        def get_params_list_from_dict(d, model_name, id=''):

            # ID used to differentiate params for a sum of models
            if id != '':
                id = '_'+id

            # Get specific smoothness and shape parameters for matern and stable
            if model_name == 'matern':
                return list([
                    d['effective_range'+id],
                    d['sill'+id],
                    d['smoothness'+id],
                    d['nugget'+id]
                ])
            elif model_name == 'stable':
                return list([
                    d['effective_range'+id],
                    d['sill'+id],
                    d['shape'+id],
                    d['nugget'+id]
                ])
            # Get nugget for only-nugget
            elif model_name == 'nugget':
                return list([d['nugget'+id]])
            # Or get classic parameters
            else:
                return list([
                    d['effective_range'+id],
                    d['sill'+id],
                    d['nugget'+id]
                ])

        # For a custom model, just pass cof in order
        if self._is_model_custom:
            list_params = self.cof
        # Get parameters for a sum of models
        elif '+' in self._model_name:
            list_model_names = self._model_name.split('+')
            list_params = []
            for i in range(len(list_model_names)):
                params = get_params_list_from_dict(d, model_name=list_model_names[i], id=str(i+1))
                list_params += params
        # Or for a single model
        else:
            list_params = get_params_list_from_dict(d, model_name=self._model_name)

        return list_params

    def to_DataFrame(self, n=100, force=False):
        """Variogram DataFrame

        Returns the fitted theoretical variogram as a pandas.DataFrame
        instance. The n and force parameter control the calculation,
        refer to the data function for more info.

        .. deprecated:: 1.0.10
            The return value of this function will change with a future release

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
        warnings.warn('The return value of this function will change in a future release.', FutureWarning)
        lags, data = self.data(n=n, force=force)

        return DataFrame({
            'lags': lags,
            self._model_name: data}
        ).copy()

    def to_gstools(self, **kwargs):
        """
        Instantiate a corresponding GSTools CovModel.

        By default, this will be an isotropic model.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to the instantiated GSTools CovModel.
            The default parameters 'dim', 'var', 'len_scale', 'nugget',
            'rescale' and optional shape parameters will be extracted
            from the given Variogram but they can be overwritten here.

        Raises
        ------
        ImportError
            When GSTools is not installed.
        ValueError
            When GSTools version is not v1.3 or greater.
        ValueError
            When given Variogram model is not supported ('harmonize').

        Warns
        -----
        Warning
            If the Variogram is a cross-variogram

        Returns
        -------
        :any:`CovModel`
            Corresponding GSTools covmodel.

        Note
        ----
        In case you intend to use the
        :func:`coordinates <skgstat.Variogram.coordinates>`
        in a GSTools workflow, you need to transpose the coordinate
        array like:

        >> cond_pos Variogram.coordinates.T

        """
        # first fit again
        self.fit()

        return skgstat_to_gstools(self, **kwargs)

    def to_gs_krige(self, **kwargs):
        """
        Instantiate a GSTools Krige class.

        This can only export isotropic models.
        Note: the `fit_variogram` is always set to `False`

        Parameters
        ----------
        variogram : skgstat.Variogram
            Scikit-GStat Variogram instamce
        **kwargs
            Keyword arguments forwarded to GSTools Krige.
            Refer to :any:`Krige <gstools.krige.Krige>` to
            learn about all possible options.
            Note that the `fit_variogram` parameter will
            always be False.

        Raises
        ------
        ImportError
            When GSTools is not installed.
        ValueError
            When GSTools version is not v1.3 or greater.
        ValueError
            When given Variogram model is not supported ('harmonize').

        Warns
        -----
        Warning
            If the Variogram is a cross-variogram

        Returns
        -------
        :any:`Krige`
            Instantiated GSTools Krige class.

        See Also
        --------
        gstools.Krige

        """
        # fit
        self.fit()
        return skgstat_to_krige(self, **kwargs)

    def plot(self, axes=None, grid=True, show=True, hist=True):
        """Variogram Plot

        Plot the experimental variogram, the fitted theoretical function and
        an histogram for the lag classes. The axes attribute can be used to
        pass a list of AxesSubplots or a single instance to the plot
        function. Then these Subplots will be used. If only a single instance
        is passed, the hist attribute will be ignored as only the variogram
        will be plotted anyway.

        .. versionchanged:: 0.4.0
            This plot can be plotted with the plotly plotting backend

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
        # get the backend
        used_backend = plotting.backend()

        if used_backend == 'matplotlib':
            return plotting.matplotlib_variogram_plot(
                self,
                axes=axes,
                grid=grid,
                show=show,
                hist=hist
            )
        elif used_backend == 'plotly':
            return plotting.plotly_variogram_plot(
                self,
                fig=axes,
                grid=grid,
                show=show,
                hist=hist
            )

        # if we reach this line, somethings wrong with plotting backend
        raise ValueError('The plotting backend has an undefined state.')

    def scattergram(self, ax=None, show=True, **kwargs):  # pragma: no cover
        """Scattergram plot

        Groups the values by lags and plots the head and tail values
        of all point pairs within the groups against each other.
        This can be used to investigate the distribution of the
        value residuals.

        .. versionchanged:: 0.4.0
            This plot can be plotted with the plotly plotting backend

        Parameters
        ----------
        ax : matplotlib.Axes, plotly.graph_objects.Figure
            If None, a new plotting Figure will be created. If given,
            it has to be an instance of the used plotting backend, which
            will be used to plot on.
        show : boolean
            If True (default), the `show` method of the Figure will be
            called. Can be set to False to prevent duplicated plots in
            some environments.

        Returns
        -------
        fig : matplotlib.Figure, plotly.graph_objects.Figure
            Resulting figure, depending on the plotting backend
        """
        # get the backend
        used_backend = plotting.backend()

        if used_backend == 'matplotlib':
            return plotting.matplotlib_variogram_scattergram(
                self,
                ax=ax,
                show=show,
                **kwargs
            )
        elif used_backend == 'plotly':
            return plotting.plotly_variogram_scattergram(
                self,
                fig=ax,
                show=show,
                **kwargs
            )

        # if we reach this line, somethings wrong with plotting backend
        raise ValueError('The plotting backend has an undefined state.')

    def location_trend(self, axes=None, show=True, **kwargs):
        """Location Trend plot

        Plots the values over each dimension of the coordinates in a scatter
        plot. This will visually show correlations between the values and any
        of the coordinate dimension. If there is a value dependence on the
        location, this would violate the intrinsic hypothesis. This is a
        weaker form of stationarity of second order.

        .. versionchanged:: 0.4.0
            This plot can be plotted with the plotly plotting backend

        Parameters
        ----------
        axes : list
            Can be None (default) or a list of matplotlib.AxesSubplots. If a
            list is passed, the location trend plots will be plotted on the
            given instances. Note that then length of the list has to match
            the dimeonsionality of the coordinates array. In case 3D
            coordinates are used, three subplots have to be given.
        show : boolean
            If True (default), the `show` method of the Figure will be
            called. Can be set to False to prevent duplicated plots in
            some environments.

        Keyword Arguments
        -----------------
        add_trend_line : bool
            .. versionadded:: 0.3.5

            If set to `True`, the class will fit a linear model to each
            coordinate dimension and output the model along with a
            calculated R². With high R² values, you should consider
            rejecting the input data, or transforming it.

            .. note::
                Right now, this is only supported for ``'plotly'`` backend


        Returns
        -------
        fig : matplotlib.Figure, plotly.graph_objects.Figure
            The figure produced by the function. Dependends on the
            current backend.

        """
        # get the backend
        used_backend = plotting.backend()

        if used_backend == 'matplotlib':
            return plotting.matplotlib_location_trend(
                self,
                axes=axes,
                show=show,
                **kwargs
            )
        elif used_backend == 'plotly':
            return plotting.plotly_location_trend(
                self,
                fig=axes,
                show=show,
                **kwargs
            )

        # if we reach this line, somethings wrong with plotting backend
        raise ValueError('The plotting backend has an undefined state.')

    def distance_difference_plot(self, ax=None, plot_bins=True, show=True):
        """Raw distance plot

        Plots all absolute value differences of all point pair combinations
        over their separating distance, without sorting them into a lag.

        .. versionchanged:: 0.4.0
            This plot can be plotted with the plotly plotting backend

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
        # get the backend
        used_backend = plotting.backend()

        if used_backend == 'matplotlib':
            return plotting.matplotlib_dd_plot(
                self,
                ax=ax,
                plot_bins=plot_bins,
                show=show
            )
        elif used_backend == 'plotly':
            return plotting.plotly_dd_plot(
                self,
                fig=ax,
                plot_bins=plot_bins,
                show=show
            )

        # if we reach this line, somethings wrong with plotting backend
        raise ValueError('The plotting backend has an undefined state.')

    def __repr__(self):  # pragma: no cover
        """
        Textual representation of this Variogram instance.

        :return:
        """
        try:
            _name = self._model_name
            _b = int(len(self.bins))
        except Exception:
            return "< abstract Variogram >"
        return "< %s Semivariogram fitted to %d bins >" % (_name, _b)

    def __str__(self):  # pragma: no cover
        """String Representation

        Descriptive representation of this Variogram instance that shall give
        the main variogram parameters in a print statement.

        Returns
        -------
        description : str
            String description of the variogram instance. Described by the
            Variogram parameters.

        """
        par = self.describe()

        _sill = np.nan if 'error' in par else par['sill']
        _range = np.nan if 'error' in par else par['effective_range']
        _nugget = np.nan if 'error' in par else par['nugget']

        s = "{0} Variogram\n".format(par['model'])
        s += "-" * (len(s) - 1) + "\n"
        s += """Estimator:         %s
        \rEffective Range:   %.2f
        \rSill:              %.2f
        \rNugget:            %.2f
        """ % (par['estimator'], _range, _sill, _nugget)

        return s
