"""
Variogram
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
    """

    """
    def __init__(self,
                 coordinates=None,
                 values=None,
                 estimator=estimators.matheron,
                 model=models.spherical,
                 dist_func='euclidean',
                 bin_func='even',
                 normalize=True,
                 fit_method='lm',
                 is_directional=False,
                 azimuth=0,
                 tolerance=45.0,
                 use_nugget=False,
                 maxlag=None,
                 n_lags=10,
                 verbose=False,
                 harmonize=False
                 ):
        """
        TODO: The cof and cov has to be reset, when: values, maxlag, fit_method

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

        # set the fitting method
        self.fit_method = fit_method

        # set directionality
        self.is_directional = is_directional
        self.azimuth = azimuth
        self.tolerance = tolerance

        # set if nugget effect shall be used
        self.use_nugget = use_nugget

        # set attributes to be filled during calculation
        self.cov = None
        self.cof = None

        # settings, not reachable by init (not yet)
        self._cache_experimental = False

        # do the preprocessing and fitting upon initialization
        self.preprocessing(force=True)
        self.fit(force=True)

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self.set_values(values=values)

    @property
    def value_matrix(self):
        return squareform(self._diff)

    def set_values(self, values):
        assert len(values) == len(self._X)
        self.cof, self.cov = None, None
        self._diff = None

        # set new values
        self._values = np.asarray(values)

        # recalculate the pairwise differences
        self._calc_diff(force=True)

    @property
    def bin_func(self):
        return self._bin_func

    @bin_func.setter
    def bin_func(self, bin_func):
        self.set_bin_func(bin_func=bin_func)

    def set_bin_func(self, bin_func):
        # reset groups and bins
        self._groups = None
        self._bins = None
        self.cof, self.cov = None, None

        if bin_func.lower() == 'even':
            self._bin_func = binning.even_width_lags
        elif bin_func.lower() == 'uniform':
            self._bin_func = binning.uniform_count_lags
        else:
            raise ValueError('%s binning method is not known' % bin_func)

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

        return self._bins

    @bins.setter
    def bins(self, bins):
        # set the new bins
        self._bins = bins

        # clean the groups as they are not valid anymore
        self._groups = None
        self.cov = None
        self.cof = None

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self.set_estimator(estimator_name=value)

    def set_estimator(self, estimator_name):
        """
        Set estimator as the new variogram estimator.

        """
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
            elif estimator_name.lower() == 'entropy':
                self._estimator = estimators.entropy
            else:
                raise ValueError(
                    'Variogram estimator %s is not understood, please' +
                    'provide the function.' % estimator_name
                )
        else:
            self._estimator = estimator_name

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

    def lag_groups(self):
        """Lag class groups

        Retuns a mask array with as many elements as self._diff has,
        identifying the lag class group for each pairwise difference. Can be
        used to extract all pairwise values within the same lag bin.

        Returns
        -------
        numpy.array

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
        for i in np.unique(self._groups):
            if i < 0:
                continue
            else:
                yield self._diff[np.where(self._groups == i)]

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
#        if force:
#            self._dist = None
#            self._diff = None
#            self._bins = None
#            self._groups = None

        # call the _calc functions
        self._calc_distances(force=force)
        self._calc_diff(force=force)
        self._calc_groups(force=force)

    def fit(self, force=False):
        """Fit the variogram

        The fit function will fit the theoretical variogram function to the
        experimental. The preprocessed distance matrix, pairwise differences
        and binning will not be recalculated, if already done. This could be
        forced by setting the force parameter to true

        Parameters
        ----------
        force : bool
            If set to True, a clean preprocessing of the distance matrix,
            pairwise differences and the binning will be forced. Default is
            False.

        Returns
        -------
        void

        """
        # delete the last cov and cof
        self.cof = None
        self.cov = None

        # if force, force a clean preprocessing
        self.preprocessing(force=force)

        # load the data
        x = self.bins
        y = self.experimental

        # remove nans
        _x = x[~np.isnan(y)]
        _y = y[~np.isnan(y)]

        if self.fit_method == 'lm':
            bounds = (0, self.__get_fit_bounds(x, y))
            self.cof, self.cov = curve_fit(self._model, _x, _y,
                                           p0=bounds[1], bounds=bounds)

        else:
            raise ValueError('Only the lm function is supported at the moment.')

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

        Returns
        -------
        callable

        """
        if self.cof is None:
            self.fit(force=True)

        # get the function
        func = self._model

        # define the wrapper
        def model(x):
            return func(x, *self.cof)

        # return
        return model

    def _calc_distances(self, force=False):
        if self._dist is not None and not force:
            return

        # else calculate the distances
        self._dist = self._dist_func(self._X)

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
                if i > j:
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
        """
        Wrapper for copy.deepcopy function of self.
        """
        return copy.deepcopy(self)

    @property
    def experimental(self):
        """

        Returns
        -------

        """
        if self.harmonize:
            return self.isotonic
        else:
            return self._experimental

    @property
    @jit
    def _experimental(self):
        """

        Returns
        -------

        """
        # prepare the result array
        y = np.zeros(len(self.bins))

        for i, lag_values in enumerate(self.lag_classes()):
            y[i] = self._estimator(lag_values)

        # apply
        return y

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
    def NRMSE(self):
        return self.rmse / np.nanmean(self.experimental)

    @property
    def NRMSE_r(self):
        return self.rmse / (np.nanmax(self.experimental) - np.nanmean(self.experimental))

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
            self.fit()

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
            range=cof[0] * maxlag,
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
            return list([d['range'], d['sill'], d['smoothness'], d['nugget']])
        elif self._model.__name__ == 'stable':
            return list([d['range'], d['sill'], d['shape'], d['nugget']])
        elif self._model.__name__ == 'nugget':
            return list([d['nugget']])
        else:
            return list([d['range'], d['sill'], d['nugget']])

    @property
    def hist(self):
        """
        Return a histogram for the present bins in the Variogram. The bin matrix bm will be unstacked and counted.
        Variogram.bins, Variogram.hist could be used to produce a properly labeled histogram.

        :return:
        """
        raise NotImplementedError

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

    def plot(self, axes=None, grid=True, show=True, cof=None):
        """
        Plot the variogram, including the experimental and theoretical variogram. By default, the experimental data
        will be represented by blue points and the theoretical model by a green line.

        TODO: REWORK this

        :return:
        """
        raise NotImplementedError
        try:
            _bin, _exp = self.bins, self.experimental
            if cof is None:
                x, data = self.data
            else:
                x = np.linspace(0, 1, 100) if self.normalized else np.linspace(0, np.nanmax(_bin), 100)
                data = self._model(x, *cof)
            _hist = self.hist
        except Exception as e:
            raise RuntimeError('A chart could not be drawn, as the input data is not complete. Please calculate the Variogram first.\nDetailed message: %s.' % str(e))

        # handle the relative experimental variogram
        if self.normalized:
            _bin /= np.nanmax(_bin)
            _exp /= np.nanmax(_exp)

        # do the plotting
        if axes is None:
            fig = plt.figure()
            ax1 = plt.subplot2grid((5, 1), (1, 0), rowspan=4)
            ax2 = plt.subplot2grid((5, 1), (0, 0), sharex=ax1)
            fig.subplots_adjust(hspace=0)
        else:
            ax1, ax2 = axes
            fig = ax1.get_figure()

        # calc histgram bar width
        # bar use 70% of the axis, w is the total width divided by amount of bins in the histogram
        w = (np.max(_bin) * 0.7) / len(_hist)

        # plot Variograms

        # if last_bin is set, plot only considered bins in blue, excluded in red
        ax1.plot(_bin, _exp, '.b')
        ax1.plot(x, data, '-g')

        if hasattr(self, 'last_bin'):
            ax1.plot(_bin[self.last_bin:], _exp[self.last_bin:], '.r')


        # plot histogram
        ax2.bar(_bin, _hist, width=w, align='center')
        # adjust
        plt.setp(ax2.axes.get_xticklabels(), visible=False)
        ax2.axes.set_yticks(ax2.axes.get_yticks()[1:])

        # ax limits
        if self.normalized:
            ax1.set_xlim([0, 1.05])
            ax1.set_ylim([0, 1.05])

        if grid:
            ax1.vlines(_bin, *ax1.axes.get_ybound(), colors=(.85, .85, .85), linestyles='dashed')
            ax2.vlines(_bin, *ax2.axes.get_ybound(), colors=(.85, .85, .85), linestyles='dashed')

        # annotation
        ax1.axes.set_ylabel('semivariance (%s)' % self._estimator.__name__)
        ax1.axes.set_xlabel('Lag (-)')
        ax2.axes.set_ylabel('N')

        # show the figure
        if show:
            fig.show()

        return fig

    def scattergram(self, ax=None, plot=True):
        """
        Plot a scattergram, which is a scatter plot of

        :return:
        """
        raise NotImplementedError
        # calculate population mean
        _mean = np.mean(self.values1D)

        # group the values to bins
        gbm = self.grouped_pairs

        # create the tail and head arrays
        tail, head = list(), list()

        # order the point pairs to tail and head
        for grp in gbm:
            # the even values are z(x) and odd values are z(x+h)
            for i in range(0, len(grp) - 1, 2):
                tail.append(_mean - grp[i])
                head.append(_mean - grp[i + 1])

        # if no plot is questioned, return tail and head arrays
        if not plot:
            return tail, head

        # else plot in either given Ax object or create a new one
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        # plot
        ax.plot(tail, head, '.k')

        # show the figure
        fig.show()

        return fig

    def location_trend(self, axes=None):
        """
        Plot the values over each dimension in the coordinates as a scatter plot.
        These plots can be used to identify a correlation between the value of an observation
        and its location. If this is the case, it would violate the fundamental geostatistical
        assumption, that a oberservation is independed of its observation

        :param axes: list of `matplotlib.AxesSubplots`. The len has to match the dimensionality
                        of the coordiantes.

        :return:
        """
        N = len(self._X[0])
        if axes is None:
            # derive the needed amount of col and row
            nrow = int(round(np.sqrt(N)))
            ncol = int(np.ceil(N / nrow))
            fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 6 ,nrow * 6))
        else:
            if not len(axes) == N:
                raise ValueError('The amount of passed axes does not fit the coordinate dimensionality of %d' % N)
            fig = axes[0].get_figure()

        for i in range(N):
            axes.flatten()[i].plot([_[i] for _ in self._X], self.values, '.r')
            axes.flatten()[i].set_xlabel('%d-dimension' % (i + 1))
            axes.flatten()[i].set_ylabel('value')

        # plot the figure and return it
        plt.tight_layout()
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
        """
        Descriptive respresentation of this Variogram instance that shall give the main variogram
        parameters in a print statement.

        :return:
        """
        par = self.describe()

        _sill = np.NaN if 'error' in par else par['sill']
        _range = np.NaN if 'error' in par else par['range']
        _nugget = np.NaN if 'error' in par else par['nugget']

        s = "{0} Variogram\n".format(par['name'])
        s+= "-" * (len(s) - 1) + "\n"
        s+= "Estimator:  {0}\nRange:      {1:.2f}\nSill:       {2:.2f}\nNugget:     {3:.2f}\n".format(par['estimator'], _range, _sill, _nugget)

        return s
