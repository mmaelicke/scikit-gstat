from functools import wraps

import numpy as np


def stvariogram(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        st = args[0]
        if st.ndim == 2:
            new_args = args[1:]
            mapping = map(lambda lags: func(lags, *new_args, **kwargs), st)
            return np.fromiter(mapping, dtype=float)
        else:
            return func(*args, **kwargs)
    return wrapper


@stvariogram
def sum(lags, Vx, Vt):
    r"""Sum space-time model

    Separable space-time variogram model. This is the most basic model as the
    two marginal models of the space and time axis are simply summed up for
    each lag pair. Further, there are no fitting parameters.
    Please consider the notes before using this model.

    Parameters
    ----------
    lags : tuple
        Tuple of the space (x) and time (t) lag given as tuple: (x, t) which
        will be used to calculate the dependent semivariance.
    Vx : skgstat.Variogram.fitted_model
        instance of the space marginal variogram with a fitted theoretical
        model sufficiently describing the marginal. If this model does not fit
        the experimental variogram, the space-time model fit will be poor as
        well.
    Vt : skgstat.Variogram.fitted_model
        instance of the time marginal variogram with a fitted theoretical
        model sufficiently describing the marginal. If this model does not fit
        the experimental variogram, the space-time model fit will be poor as
        well.

    Returns
    -------
    gamma : float
        The semi-variance modeled for the given lags.

    Notes
    -----
    This model is implemented like:

    .. math::
        \gamma (h,t) = \gamma_x (h) + \gamma_t (t)
    
    Where :math:`\gamma_x(h)` is the spatial marginal variogram and 
    :math:`\gamma_t(t)` is the temporal marginal variogram.

    It is not a good idea to use this model in almost any case, as it assumes
    the covariance field to be isotropic in space and time direction,
    which will hardly be true. Further, it might not be strictly definite as
    shown by [7]_, [8]_, [9]_.

    References
    ----------
    .. [7] Myers, D. E., Journel, A. (1990), Variograms with Zonal
       Anisotropies and Non-Invertible Kriging Systems.
       Mathematical Geology 22, 779-785.
    .. [8] Dimitrakopoulos, R. and Lou, X. (1994), Spatiotemporal modeling:
       covariances and ordinary kriging systems, in R. Dimitrakopoulos,
       (ed.) Geostatistics for the next century, Kluwer Academic Publishers,
       Dodrecht 88-93.

    """
    h, t = lags
    return Vx(h) + Vt(t)


@stvariogram
def product(lags, Vx, Vt, Cx, Ct):
    r"""Product model

    Separable space-time variogram model. This model is based on the product
    of the marginal space and time models.

    Parameters
    ----------
    lags : tuple
        Tuple of the space (x) and time (t) lag given as tuple: (x, t) which
        will be used to calculate the dependent semivariance.
    Vx : skgstat.Variogram.fitted_model
        instance of the space marginal variogram with a fitted theoretical
        model sufficiently describing the marginal. If this model does not fit
        the experimental variogram, the space-time model fit will be poor as
        well.
    Vt : skgstat.Variogram.fitted_model
        instance of the time marginal variogram with a fitted theoretical
        model sufficiently describing the marginal. If this model does not fit
        the experimental variogram, the space-time model fit will be poor as
        well.
    Cx : float
        Marginal space sill.
    Ct : float
        Marignal time sill.

    Returns
    -------
    gamma : float
        The semi-variance modeled for the given lags.

    Notes
    -----
    The product sum model is implemented following [14]_:

    .. math::
        \gamma (h,t) = C_x * \gamma_t(t) + C_t * \gamma_x(h) - \gamma_x(h) * \gamma_t(t) 
    
    Where :math:`\gamma_x(h)` is the spatial marginal variogram and 
    :math:`\gamma_t(t)` is the temporal marginal variogram.

    References
    ----------
    .. [14] De Cesare, L., Myers, D., and Pose, D. (201b), FORTRAN 77 programs
       for space-time modeling, Computers & Geoscience 28, 205-212.

    """
    h, t = lags
    return Cx * Vt(t) + Ct * Vx(h) - Vx(h) * Vt(t)


@stvariogram
def product_sum(lags, Vx, Vt, k1, k2, k3, Cx, Ct):
    r"""Product-Sum space-time model

    Separable space-time variogram model, based on a combination of 'sum' and
    'product' models. Both base models are based on separated marginal
    variograms for the space and time axis.

    Parameters
    ----------
    lags : tuple
        Tuple of the space (x) and time (t) lag given as tuple: (x, t) which
        will be used to calculate the dependent semivariance.
    Vx : skgstat.Variogram.fitted_model
        instance of the space marginal variogram with a fitted theoretical
        model sufficiently describing the marginal. If this model does not fit
        the experimental variogram, the space-time model fit will be poor as
        well.
    Vt : skgstat.Variogram.fitted_model
        instance of the time marginal variogram with a fitted theoretical
        model sufficiently describing the marginal. If this model does not fit
        the experimental variogram, the space-time model fit will be poor as
        well.
    k1 : float
        Fitting parameter. k1 has to be positive or zero and may not be larger
        than all marginal sill values.
    k2 : float
        Fitting paramter. k2 has to be positive or zero and may not be larger
        than all marginal sill values.
    k3 : float
        Fitting parameter. k3 has to be positive and may not be larger than
        all marginal sill values.
    Cx : float
        Marginal space sill.
    Ct : float
        Marignal time sill.

    Returns
    -------
    gamma : float
        The semi-variance modeled for the given lags.

    Notes
    -----
    This model implements the product-sum model as suggested by
    De Cesare et. al [15]_, [16]_:

    .. math::
        \gamma_{ST}(h_s, h_t) = [k_1C_T(0) + k_2]*\gamma_S(h_s) +
        [k_1C_s(0) + k_3]\gamma_T(h_t) - k_1\gamma_s(h_s) x \gamma_T(h_t)

    References
    ----------
    .. [15] De Cesare, L., Myers, D. and Posa, D. (2001a), Product-sum
       covariance for space-time mdeling, Environmetrics 12, 11-23.
    .. [16] De Cesare, L., Myers, D., and Pose, D. (201b), FORTRAN 77 programs
       for space-time modeling, Computers & Geoscience 28, 205-212.

    """
    h, t = lags
    return (k2 + k1*Ct)*Vx(h) + (k3 + k1*Cx) * Vt(t) - k1 * Vx(h) * Vt(t)
