try:
    import pykrige
    PYKRIGE_AVAILABLE = True
except ImportError:
    PYKRIGE_AVAILABLE = False

import numpy as np


def __check_pykrige_available():  # pragma: no cover
    if not PYKRIGE_AVAILABLE:
        print('The pykrige interface needs pykrige installed.')
        print("Run 'pip install pykrige' to install it.")
        return False
    return True


def pykrige_model(variogram):
    """
    """
    # pykrige is available?
    if not __check_pykrige_available():
        return

    # get the fitted model
    model = variogram.fitted_model

    # define the model function
    def skgstat_model(parameters, lags):
        """Variogram model

        This function is a interface from scikit-gstat to pykrige.
        If you want to use a fitted skgstat.Variogram instance as a
        custom variogram model in pykrige, this is the already fitted
        function that can be passed as the `variogram_function` argument.
        Additionally, you need to set the `variogram_model` to `'custom'`.

        The skgstat.interfaces.pykrige also has a pykrige_as_kwargs
        function. That will return all necessary keyword arguments for the
        pykrige class as a dictionary. You can just pass it using the double
        star operator.

        """
        if not isinstance(lags, np.ndarray):
            lags = np.asarray(lags, dtype=float)

        # get the semi-variances
        semivar = np.fromiter(map(model, lags.flatten()), dtype=float)

        # return
        return semivar.reshape(lags.shape)

    # return model
    return skgstat_model


def pykrige_params(variogram):
    """
    """
    # pykrige is available?
    if not __check_pykrige_available():
        return

    # get the parameters into the correct order. 
    pars = variogram.parameters

    return [pars[1], pars[0], pars[2]]


def pykrige_as_kwargs(variogram, adjust_maxlag=False, adjust_nlags=False):
    """
    """
    # pykrige is available?
    if not __check_pykrige_available():
        return

    # as far as I get it, there is no maximum lag in pykrige. 
    if adjust_maxlag:
        variogram.maxlag = None
    else:
        print('[WARNING]: If the maximum lag is not None, the variogram plots will differ.')

    # to work properly, variogram has to use a nugget
    variogram.use_nugget = True
    variogram.fit()

    # get the model
    model_func = pykrige_model(variogram)

    # get the parameters
    pars = pykrige_params(variogram)

    args = dict(
        variogram_model='custom',
        variogram_parameters=pars,
        variogram_function=model_func
    )

    if adjust_nlags:
        args['nlags'] = variogram.n_lags

    # return
    return args
