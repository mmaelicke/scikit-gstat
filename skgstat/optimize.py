"""
This module defines optimization classes for variograms. The class can be used to test all variogram combinations
on a dataset and evaluate by the given model fit parameter.
"""
from .distance import nd_dist
from skgstat import binify_even_width, Variogram
import numpy as np
import sys, itertools, warnings
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


class VariogramFitter:
    """

    """

    def __init__(self, coordinates=None, values=None, verbose=True, eval='RMSE', autorun=False,
                 dm=None, bm=None, normalize=True, fit_method='lm', pec_punish=1.0,
                 dm_func=nd_dist, bm_func=binify_even_width, is_directional=False, azimuth=0, tolerance=45.0,
                 use_nugget=[False, True], estimator=['matheron', 'cressie', 'dowd'],
                 model=['spherical', 'exponential', 'gaussian', 'cubic', 'stable', 'matern'], N=[8, 10, 15, 20],
                 maxlag=[0.1, 0.25, 0.3, 0.5, 0.75]):

        # set verbosity and evaluation parameter
        self.verbose=verbose
        self.eval=eval

        # set the static parameters
        self.static = dict(
            coordinates=coordinates,
            values=values,
            dm=dm,
            bm=bm,
            normalize=normalize,
            fit_method=fit_method,
            pec_punish=pec_punish,
            dm_func=dm_func,
            bm_func=bm_func,
            is_directional=is_directional,
            azimuth=azimuth,
            tolerance=tolerance
        )

        # check the optimization parameters for data tyoe
        # if list or tuple, optimize, else add to static
        self.params = dict(
            use_nugget=use_nugget,
            estimator=estimator,
            model=model,
            N=N,
            maxlag=maxlag
        )

        # move all params with only one option to the static dict
        keys = list(self.params.keys())
        for k in keys:
            if not isinstance(self.params[k], (list, tuple)):
                # this has only one option
                self.static[k] = self.params[k]
                del self.params[k]

            elif len(self.params[k]) == 1:
                self.static[k] = self.params[k][0]
                del self.params[k]

        # combine the params
        self.names = list(self.params.keys())
        self.combinations = list(itertools.product(*self.params.values()))
        self.n = len(self.combinations)

        # build the result container
        self.V = list()
        self.e = list()
        self.error = list()

        if autorun:
            self.Variogram()

    def run(self):
        """

        :return:
        """
        # reset the result container
        self.V = list()
        self.e = list()
        self.error = list()

        for i, combination in enumerate(self.combinations):
            self.__build_variogram(combination=combination)

            if self.verbose:
                sys.stdout.write('%d/%d (%.1f%%) calculated.\n' % (i, self.n, (i+1) / self.n *100))

    def Variogram(self):
        """

        :return:
        """
        self.run()

        # find the best Variogram
        idx = np.nanargmin(self.e)

        return self.V[idx]

    def plot(self, path):
        """

        :param path:
        :return:
        """
        self.run()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with PdfPages(path) as pdf:
                for i, tup in enumerate(zip(self.V, self.e)):
                    V, e = tup
                    if V is not None:
                        V.plot();
                        plt.subplots_adjust(right=0.7)
                        plt.figtext(0.73, 0.73, '%s: %.2f' % (self.eval, e), fontsize=14)
                        plt.figtext(0.73, 0.45, str(V), fontsize=10)
                        plt.figtext(0.73, 0.29, '\n'.join(['%s: %s' % (n, str(v)) for n,v in zip(self.names, self.combinations[i]) ]), fontsize=10)
                        pdf.savefig(plt.gcf())
                    else:
                        f,ax = plt.subplot(1,1)

                        plt.figtext(1, 0.73, 'No Result for combination: ', fontsize=14)
                        plt.figtext(0.1, 0.45, '\n'.join(['%s: %s' % (n, str(v)) for n,v in zip(self.names, self.combinations[i]) ]), fontsize=10)
                        pdf.savefig()

                    if self.verbose:
                        sys.stdout.write('%d/%d plots drawn.\n' % (i+1, self.n))


    def __build_variogram(self, combination):
        """
        Return the Variogram of the given combination. This combination tuple has to align with the self.names property

        :param combination:
        :return:
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                # build the variogram
                V = Variogram(**self.static, **{n:v for n,v in zip(self.names, combination)})

                # evaluate the result
                e = getattr(V, self.eval)

            except Exception as err:
                V = None
                e = np.NaN
                self.error.append([combination, str(err)])
            finally:
                # append
                self.V.append(V)
                self.e.append(e)