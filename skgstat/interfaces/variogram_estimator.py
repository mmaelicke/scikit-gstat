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
                 use_score='rmse',
                 cross_validate=False,
                 **kwargs
                 ):
        r"""VariogramEstimator class

        Interface class for usage with scikit-learn. This class is intentended
        for usage with the GridSearchCV or Pipeline classes of scikit-learn.

        The input parameters are the same as for the
        :class:`Variogram <skgstat.Variogram>` class.
        Refer to the documentation there.

        The only parameter specific to the Estimator class is the `use_score`
        attribute. This can be the root mean squared error (rmse), mean squared
        error (mse) or mean absoulte error (mae). The Estimater can either calculate
        the score based on the model fit (model ~ experimental) or using a
        leave-one-out cross-validation of a OrdinaryKriging using the model

        .. versionchanged:: 0.5.4
            Uses ['rmse', 'mse', 'mae'] as scores exclusesively now.
            Therefore, either the fit of the Variogram or a cross validation
            can be used for scoring

        Parameters
        ----------
        use_score : str
            Scoring parameter to assess the Variogram fitting quality.
            Defaults to `'rmse'`, the Root mean squared error.
            Can be changed to `['mse', 'mae']`.
        cross_validate : bool
            .. versionadded:: 0.5.4
            If True, the score will be calculate from a cross-validation of
            the variogram model in OrdinaryKriging, rather than the model fit.

        Keyword Arguments
        -----------------
        cross_n : int
            .. versionadded:: 0.5.4
            If not None, this is the amount of points (and iterations) used in
            cross valiation. Does not have any effect if `cross_validate=False`.
        seed : int
            .. versionadded:: 0.5.4
            Will be passed down to the
            :func:`cross_validation <skgstat.Variogram.cross_validate>`
            method of Variogram.

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
        self.cross_validate = cross_validate
        self._kwargs = kwargs

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

    def score(self, X=None, y=None):
        """Fit score
        .. versionchanged:: 0.5.4
            Can now use cross-validated scores

        Score ('rmse', 'mse', 'mae') based on the fitting.

        """
        if self.cross_validate:
            # check if a n was given
            n = self._kwargs.get('cross_n')
            return self.variogram.cross_validate(
                n=n,
                metric=self.use_score,
                seed=self._kwargs.get('seed')
            )
        else:
            # return the score
            return getattr(self.variogram, self.use_score)
