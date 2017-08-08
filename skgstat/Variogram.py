"""
Variogram Klasse
"""

from skgstat.distance import nd_dist
from skgstat.binning import binify_even_width, binify_even_bin, group_to_bin
from skgstat.estimator import matheron, cressie, dowd, genton, minmax, entropy
from skgstat.models import spherical, exponential, gaussian, cubic, stable, matern
import numpy as np
from pandas import DataFrame
import copy
import matplotlib.pyplot as plt
from matplotlib.axes import SubplotBase
from scipy.optimize import curve_fit


class Variogram(object):
    """
    Variogram repesentation.

    This class can generate a Semivariogram from a given sample.
    By default, the sample point pairs are ordered into even-width bin,
    separated by the euclidean distance of the point pairs.
    The Semivariance in the bin is calculated by the Matheron estimator
    and a spherical Varigram function is fitted by least squares to the experimental Variogram.

    The distance matrix, bin matrix, semi-variance estimator and theoretical variogram function can all be changed on
    instantiation. The Variogram class can be used to return the result data, plot the Variogram, estimate the
    kriging weights and create a Kriging instance.

    """
    def __init__(self, coordinates=None, values=None, dm_func=nd_dist, bm_func=binify_even_width,
                 estimator=matheron, model=spherical, dm=None, bm=None, normalize=True, fit_method='lm',
                 pec_punish=1.0, is_directional=False, azimuth=0, tolerance=45.0, use_nugget=False, maxlag=None,
                 N=None, verbose=False):
        """

        :param coordinates: numpy array or list with the coordinates of the sample as tuples
        :param values: numpy array or list with the Values of the sample. If ndim > 1 an aggregator is used
        :param dm_func: function which is used to calculate the distance matrix
        :param bm_func: function which is used to calculate the binning matrix
        :param estimator: estimator can be a function or a string identifying a standard estimator
               Supported are 'matheron', 'cressie', 'dowd' or 'genton'
        :param model: string or callable with the theoretical variogram function
        :param dm: numpy array with the distance matrix of the given sample
        :param bm: numpy array with the binning matrix of the given sample
        :param normalize: boolean, specify if the lag should be given absolute or relative to the maxlag
        :param fit_method: The method for fitting the theoretical model.
               Either 'lm' for least squares or 'pec' for point exclusion cost'
        :param pec_punish: If 'pec' is the fit_method, give the power of punishing the point exclusion.
               1 is full punish; 0 non-punish.
        :param is_directional:
        :param azimuth
        :param tolerance:
        :param use_nugget: boolean, set if nugget effect shall be used
        :param maxlag:
        :param N: number of bins
        :param verbose:
        """

        # Set coordinates and values
        self._X = list(coordinates)
        self.values = list(values)

        # set verbosity
        self.verbose = verbose

        # if values is given with ndim > 1, set an aggregator
        self.agg = np.nanmean

        # bm properites
        self.maxlag = maxlag
        self._bm_kwargs = {}
        self._dm_kwargs = {}

        # save the functions, if the matrixes are not given
        if dm is None:
            self.dm_func = dm_func
        else:
            self._dm = dm

        if bm is None:
            self.bm_func = bm_func
        else:
            self._bm = bm

        # estimator can be a function or a string identifying a standard estimator
        self.set_estimator(estimator=estimator)

        # model can be a function or a string identifying a standard variogram function
        self.set_model(model=model)

        # specify if the lag should be given absolute or relative to the maxlag
        self.normalized = normalize

        # set the fitting method and model quality measure
        self.fit_method = fit_method
        self.pec_punish = pec_punish

        # set directionality
        self.is_directional = is_directional
        self.azimuth = azimuth  # Set is_directional as True if azimuth is given?
        self.tolerance = tolerance

        # set if nugget effect shall be used
        self.use_nugget = use_nugget

        # set binning matrix if N was given
        if N is not None:
            self.set_bm(N=N)

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self.set_estimator(estimator=value)

    def set_estimator(self, estimator):
        """
        Set estimator as the new Variogram estimator. Either a function returning a single or a list of semivariance
        values is needed, or a string identifiying a default one.
        Supported are 'matheron', 'cressie', 'dowd' or 'genton'.
        """
        if isinstance(estimator, str):
            if estimator.lower() == 'matheron':
                self._estimator = matheron
            elif estimator.lower() == 'cressie':
                self._estimator = cressie
            elif estimator.lower() == 'dowd':
                self._estimator = dowd
            elif estimator.lower() == 'genton':
                self._estimator = genton
            elif estimator.lower() == 'minmax':
                self._estimator = minmax
            elif estimator.lower() == 'entropy' or estimator.lower() == 'h':
                self._estimator = entropy
            else:
                raise ValueError('Variogram estimator %s is not understood, please provide the function.' % estimator)
        else:
            self._estimator = estimator

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self.set_model(model=value)

    def set_model(self, model):
        """
        Set model as the new theoretical variogram function. Either a function returning the semivariance at a given lag
        for given parameters is needed, or a string identifying a default one. Supported are 'spherical', 'exponential'
        or 'gaussian'.

        :param model:
        :return:
        """
        if isinstance(model, str):
            if model.lower() == 'spherical':
                self._model = spherical
            elif model.lower() == 'exponential':
                self._model = exponential
            elif model.lower() == 'gaussian':
                self._model = gaussian
            elif model.lower() == 'cubic':
                self._model = cubic
            elif model.lower() == 'stable':
                self._model = stable
            elif model.lower() == 'matern':
                self._model = matern
            else:
                raise ValueError('The theoretical Variogram function %s is not understood, '
                                 'please provide the function' % model)
        else:
            self._model = model

    def binify_like(self, how='even width'):
        """
        Specify, how the bins for the Variogram shall be drawn. how can identify one of the prepared binning algorithms.
        Use 'even_width' or 'even width' for the binify_even_width function and 'even bin' or 'even_bin' for the
        binity_even_bins function. If how is callable, it will be used as bm_func.
        It will be called by the bm property with N, the number of bins and the bm_kwargs as arguments and has to return
        the bm matrix and an array of bin widths.

        :param how:
        :return:
        """
        # remove the _bm if necessary
        if hasattr(self, '_bm'):
            delattr(self, '_bm')
        if hasattr(self, 'bin_widths'):
            delattr(self, 'bin_widths')

        if callable(how):
            self.bm_func = how
        elif how.lower().replace('_', ' ') == 'even width':
            self.bm_func = binify_even_width
        elif how.lower().replace('_', ' ') == 'even bin':
            self.bm_func = binify_even_bin
        else:
            raise ValueError("how has to be a callable or one of ['even width', 'even bin']")

    def clone(self):
        """
        Wrapper for copy.deepcopy function of self.
        """
        return copy.deepcopy(self)

    def set_dm(self, dm=None, **kwargs):
        """
        calculates the distance matrix if needed and sets it as attribute self._dm
        """
        # update kwargs
        self._dm_kwargs.update(kwargs)

        if dm is None:
            self._dm = self.dm_func(self._X, **self._dm_kwargs)
            if hasattr(self, '_bm'):
                self.set_bm()
        else:
            self._dm = dm

    @property
    def dm(self):
        """

        :return:
        """
        if hasattr(self, '_dm'):
            return self._dm
        else:
            return self.dm_func(self._X, **self._dm_kwargs)


    def set_bm(self, bm=None, maxlag=None, **kwargs):
        """

        :return:
        """
        # overwrite maxlag
        if maxlag is not None:
            if maxlag < 1:
                self.__maxlag = maxlag * np.max(self.dm)
            else:
                self._maxlag = maxlag

        # store the bm_kwargs
        self._bm_kwargs.update(kwargs)

        if bm is None:
            self._bm, self.bin_widths = self.bm_func(X=self._X, dm=self.dm, maxlag=self.maxlag, **self._bm_kwargs)
        else:
            if hasattr(self, 'bin_widths'):
                delattr(self, 'bin_widths')
            self._bm = bm

    @property
    def bm(self):
        """

        :return:
        """
        if hasattr(self, '_bm'):
            return self._bm
        else:
            _bm, self.bin_widths = self.bm_func(X=self._X, dm=self.dm, maxlag=self.maxlag, **self._bm_kwargs)
            return _bm

    @property
    def maxlag(self):
        return self._maxlag

    @maxlag.setter
    def maxlag(self, value):
        # a maxlag was set, therefore the _bm and bin_widths attributes have to be cleared
        if hasattr(self, '_bm'):
            delattr(self, '_bm')
        if hasattr(self, 'bin_widths'):
            delattr(self, 'bin_widths')

        # set new maxlag
        if value is None:
            self._maxlag = None
        elif value < 1:
            self._maxlag = value * np.max(self.dm)
        else:
            self._maxlag = value

    @property
    def experimental(self):
        """
        :return: experimental variogram as a numpy.ndarray.
        """
        # group the values to bins and apply the estimator
        _g = self.grouped_pairs

        # apply
        return np.array(self._estimator(_g))

    @property
    def grouped_pairs(self):
        """
        Result of the group_to_bin function. This property will be used for wrapping the function, in case there are
        alternative grouping functions implemented one day.

        :return:
        """
        if self.is_directional:
            return group_to_bin(self.values, self.bm, X=self._X, azimuth_deg=self.azimuth, tolerance=self.tolerance,
                                maxlag=self.maxlag)
        else:
            return group_to_bin(self.values, self.bm, maxlag=self.maxlag)


    def fit(self, x, y):
        """
        Fit the theoretical variogram function to the experimental values. The function will be fitted using the least
        square method, with a maximum of maxiter iteration, defaults to 1000. For fitting the starting parameters of
        the theoretical function a, C0, b can be given as kwargs. If the Variogram uses a custom model, the parameters
        might have other names.
        The parameter set with best fit will be returned.

        :return:
        """
        # if last bin is set, delete it
        if hasattr(self, 'last_bin'):
            del self.last_bin

        # remove nans
        _x = x[~np.isnan(y)]
        _y = y[~np.isnan(y)]

        if self.fit_method == 'lm':
#            bounds = (0, [np.nanmax(x), np.nanmax(y)])
            bounds = (0, self.__get_fit_bounds(x, y))
            return curve_fit(self._model, _x, _y, p0=bounds[1], bounds=bounds)

        elif self.fit_method == 'pec':
            # Run the fitting with each point excluded
            rmse, cof, cov, punish = list(), list(), list(), list()
            # get the histogram counts (the cumsum of bin sizes, summed from right to left)
            _h = self.hist[~np.isnan(y)]
            h = np.flipud(np.cumsum(np.flipud(_h)))

            for i in range(1, len(_x) - 2):
#                bnd = (0, [np.max(_x[:-i]), np.max(_y[:-i])])
                bnd = (0, self.__get_fit_bounds(x[:-i], y[:-i]))
                cf, cv = curve_fit(self._model, _x[:-i], _y[:-i], p0=bnd[1], bounds=bnd)
                cof.append(cf)
                cov.append(cov)
                p = ((h[0] + 1) / ((h[0] + 1) - h[-i]))**self.pec_punish
                rmse.append(p * (np.sqrt(np.sum((self._model(_x[:-i], *cf) - _x[:-i]) ** 2) / len(_x[:-i]))))
                punish.append(p)

            if self.verbose:
                print('Punish Weights: ', ['%.3f' % _ for _ in punish])
                print('RMSE: ', ['%.2f' % _ for _ in rmse])
            # here rmse is the error for the cf set in cof
            # find the optimum of excluded points to rmse error improvement
            best_rmse = rmse.index(np.nanmin(rmse))
            self.last_bin = len(x) - (best_rmse + 1)

            return cof[best_rmse], cov[best_rmse]

        else:
            raise ValueError('The fit method {} is not understood. Use either \'lm\' (least squares) or \'pec\' (point exclusion cost).'.format(self.fit_method))


    def __get_fit_bounds(self, x, y):
        """
        Return the bounds for parameter space in fitting a variogram model. The bounds are depended on the Model
        that is used

        :return:
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

    @property
    def data(self):
        """
        Calculates the experimental Variogram. As the bins are only indexed by a integer, the lag array is caculated
        by cummulative summarizing the bin_widths array. If this bin_widths was not returned by the bm_func, even bin
        widths are created matching the number of bins at the maximum entry in the distance matrix. This will lead to
        calculation errors, in case the bin widths are not even.
        The theoretical variogram function is fitted to the experimental values at given lags
        and the theoretical function is calculated for all increments and returned.

        :return:
        """
        # calculate the experimental variogram
        # this might raise an exception if the bm is not present yet
        _exp = self.experimental
        _bin = self.bins

        # use relative or absolute bins
        if self.normalized:
            _bin /= np.nanmax(_bin)     # normalize X
            _exp /= np.nanmax(_exp)     # normalize Y
            x = np.linspace(0, 1, 100)  # use 100 increments
        else:
#            x = np.arange(0, np.float64(np.max(_bin)), 1)
            x = np.linspace(0, np.float64(np.nanmax(_bin)), 100)

        # fit
        self.cof, self.cov = self.fit(_bin, _exp)

        return x, self._model(x, *self.cof)

    @property
    def residuals(self):
        """

        :return:
        """
        # get the deviations
        experimental, model = self.__model_deviations()

        # calculate the residuals
        return np.fromiter(map(lambda x, y: x - y, model, experimental), np.float)

    @property
    def mean_residual(self):
        """

        :return:
        """
        return np.nanmean(np.fromiter(map(np.abs, self.residuals), np.float))

    @property
    def RMSE(self):
        # get the deviations
        experimental, model = self.__model_deviations()

        # get the sum of squares
        rsum = np.nansum(np.fromiter(map(lambda x, y: (x - y)**2, experimental, model), np.float))

        return np.sqrt(rsum / len(model))

    @property
    def NRMSE(self):
        return self.RMSE / np.nanmean(self.experimental)

    @property
    def NRMSE_r(self):
        return self.RMSE / (np.nanmax(self.experimental) - np.nanmean(self.experimental))

    @property
    def r(self):
        """
        Pearson correlation of the fitted Variogram

        :return:
        """
        # get the experimental and theoretical variogram and cacluate means
        experimental, model = self.__model_deviations()
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
        experimental, model = self.__model_deviations()
        mx = np.nanmean(experimental)

        # calculate the single nash-sutcliffe terms
        term1 = np.nansum(np.fromiter(map(lambda x, y: (x - y)**2, experimental, model), np.float))
        term2 = np.nansum(np.fromiter(map(lambda x: (x - mx)**2, experimental), np.float))

        return 1 - (term1 / term2)

    def __model_deviations(self):
        """
        These model deviations can be used to calculate different model quality parameters like residuals, RMSE.
        :return:
        """
        # get the experimental values and their bin bounds
        _exp = self.experimental
        _bin = self.bins

        # get the model parameters
        param = self.describe()
        if 'error' in param:
            raise RuntimeError('The Variogram cannot be applied properly. First, calculate the variogram.')

        # calculate the model values at bin bounds
        _model = self._model(_bin, *self.cof)

        return _exp, _model

    def describe(self):
        """
        Return a dictionary of the Variogram parameters

        :return:
        """
        try:
            if self.normalized:
                maxlag = np.nanmax(self.bins)
                maxvar = np.nanmax(self.experimental)
            else:
                maxlag = 1
                maxvar = 1
            index, data = self.data
            cof = self.cof
        except:
            return dict(name=self._model.__name__, estimator = self._estimator.__name__, error='Variogram not calculated.')
        rdict = dict(
            name=self._model.__name__,
            estimator=self._estimator.__name__,
            range=cof[0] * maxlag,
            sill=cof[1] * maxvar,
            nugget=cof[-1] if self.use_nugget else 0
        )

        # handle s parameters
        if self._model.__name__ == 'matern':
            rdict['smoothness'] = cof[2]
        elif self._model.__name__ == 'stable':
            rdict['shape'] = cof[2]

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
    def bins(self):
        """
        Return the upper bin bounds of the experimental Variogram.
        If no self.bin_widths is set, even width bins are assumed and will be calculated from the distance matrix,
        as the maximum distance divided by the amount of bins.

        This will cause errors if the bins are not evenly distributed.

        :return:
        """
        # do a dummy bm calculation to set actual bin widths
        # TODO: this is just a ugly workaround.
        _bm = self.bm

        if hasattr(self, 'bin_widths'):
            _bin = np.cumsum(self.bin_widths)
        else:
            print('Warning: Bin edges were calcuated on the fly.')
            n = int(np.max(self.bm) + 1)
            _bin = np.cumsum(np.ones(n) * np.max(self.dm) /  n)

        return _bin

    @property
    def hist(self):
        """
        Return a histogram for the present bins in the Variogram. The bin matrix bm will be unstacked and counted.
        Variogram.bins, Variogram.hist could be used to produce a properly labeled histogram.

        :return:
        """
        # get the upper triangle from the bin matrix
#        _bm = self.bm
#        ut = [_bm[i, j] for i in range(len(_bm)) for j in range(len(_bm)) if j > i]
#        return np.bincount(ut)

        # get the grouped pairs
        _g = self.grouped_pairs
        return np.array([len(_) / 2 for _ in _g])

    @property
    def values1D(self):
        """
        If the values were given with ndim > 1, the value1D property will return the aggreagtes
        using the self.agg function.

        :return:
        """
        return [self.agg(_) for _ in self.values]

    def to_DataFrame(self):
        """
        Return the result of the theoretical Variogram as a pandas.DataFrame. The lag will index the semivariance
        values.

        :return:
        """
        index, data = self.data

        # translate the normalized lag to real lag
        maxlag = np.nanmax(self.bins)
        maxvar = np.nanmax(self.experimental)

        return DataFrame({'lag': index * maxlag, self._model.__name__: data * maxvar}).copy()


    def plot(self, axes=None, grid=True, show=True, cof=None):
        """
        Plot the variogram, including the experimental and theoretical variogram. By default, the experimental data
        will be represented by blue points and the theoretical model by a green line.

        :return:
        """
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


    ## ----- implementing some Python functions ---- ##
    def __repr__(self):
        """

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
