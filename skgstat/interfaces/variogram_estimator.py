import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y


class VariogramEstimator(BaseEstimator):
    def __init__(self,
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
                 use_score='rmse'
                 ):
        r"""VariogramEstimator class

        Interface class for usage with scikit-learn. This class is intentended
        for usage with the GridSearchCV or Pipeline classes of scikit-learn.

        The input parameters are the same as for the
        :class:`Variogram <skgstat.Variogram>` class.
        Refer to the documentation there.

        The only parameter specific to the Estimator class is the `score`
        attribute.

        Parameters
        ----------
        score : str
            Scoring parameter to assess the Variogram fitting quality.
            Defaults to `'rmse'`, the Root mean squared error.
            Can be changed to ``['r2', 'residuals']``.

        Note
        ----
        The workflow of this class is a bit different from the Variogram class.
        The Variogram parameters are passed on instantiation. The actual data,
        coordinates and values, are then passed to the fit method, which
        returns a fitted instance of the model. The predict method takes
        **distance** values and *predicts* the semi-variance according to the
        fitted model. This is in line with the Estimators of sklearn, but
        breaks the guidelines in one point, as the X passed to fit and
        predict are in fact two different things (and of different shape).

        """
        # store all the passed attributes.
        # they will be needed to create the Variogram
        self.estimator = estimator
        self.model = model
        self.dist_func = dist_func
        self.bin_func = bin_func
        self.normalize = normalize
        self.fit_method = fit_method
        self.fit_sigma = fit_sigma
        self.use_nugget = use_nugget
        self.maxlag = maxlag
        self.n_lags = n_lags
        self.verbose = verbose

        # add Estimator specific attributes
        self.use_score = use_score

        # This is a workaround due to circular imports
        from skgstat import Variogram
        self.VariogramCls = Variogram

    def fit(self, X, y):
        """Fit a model

        Fits a variogram to the given data.

        Parameters
        ----------
        X : numpy.ndarray
            input data coordinates. Usually 2D or 3D data,
            but any dimensionality is allowed.
        y : numpy.ndarray
            observation values at the location given in X.
            Has to be one dimensional

        Returns
        -------
        variogram : VariogramEstimator
            A fitted instance of VariogramEstimator

        """
        # check the input data
        X, y = check_X_y(X, y)

        # build the model
        self.variogram = self.VariogramCls(
            X, y,
            estimator=self.estimator,
            model=self.model,
            dist_func=self.dist_func,
            bin_func=self.bin_func,
            normalize=self.normalize,
            fit_method=self.fit_method,
            fit_sigma=self.fit_sigma,
            use_nugget=self.use_nugget,
            maxlag=self.maxlag,
            n_lags=self.n_lags
        )

        # append the data
        self.X_ = X
        self.y_ = y

        # get the fitted model function
        self._model_func_ = self.variogram.fitted_model

        # append the variogram parameters
        d = self.variogram.describe()
        self.range_ = d['effective_range']
        self.sill_ = d['sill']
        self.nugget_ = d['nugget']

        # return
        return self

    def predict(self, X):
        """Predict

        Predicting function. A prediction in this context is
        the estimation of semi-variance values for a given distance
        array. The X here is an 1D array of distances, **not coordinates**.

        """
        return np.fromiter(map(self._model_func_, X.flatten()), dtype=float)

    def score(self, X, y=None):
        """Fit score

        Score based on the fitting.

        """
        # TODO: maybe I have to create a new V, fitted to X,y here?

        # return the score
        return getattr(self.variogram, self.use_score)
