try:
    import gstools
    GSTOOLS_AVAILABLE = True
except ImportError:
    GSTOOLS_AVAILABLE = False
import numpy as np


def __check_gstools_available():  # pragma: no cover
    if not GSTOOLS_AVAILABLE:
        print('The gstools interface needs gstools installed.')
        print("Run 'pip install gstools' to install it.")
        return False
    return True


def gstools_cov_model(variogram, **kwargs):
    """GSTools Interface

    Pass a :class:`skgstat.Variogram` instance.
    Returns an **already fitted** variogram model
    inherited from `gstools.CovModel`. All
    kwargs passed will be passed to
    `gstools.CovModel`.

    """
    # extract the fitted variogram model
    fitted_model = variogram.fitted_model

    # define the CovModel
    class VariogramModel(gstools.CovModel):
        def variogram(self, r):
            if isinstance(r, np.ndarray):
                return fitted_model(r.flatten()).reshape(r.shape)
            else:
                return fitted_model(r)

    # dim can be infered from variogram
    if 'dim' not in kwargs.keys():
        kwargs['dim'] = variogram.coordinates.ndim

    # Create the instance
    model = VariogramModel(**kwargs)
    model.fit_variogram(variogram.bins, variogram.experimental)

    return model
