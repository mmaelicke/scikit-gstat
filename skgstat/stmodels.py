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
        Fitting parameter. k2 has to be positive or zero and may not be larger
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


def metric(lags, model, rx, rt, *args):
    r"""Metric space-time model

    Non-separable space-time variogram model. The time axis is rescaled
    into a space distance and one variogram model is applied to the
    combined space-time distance. This is a geometric anisotropy between
    space and time.

    .. versionadded:: 1.0.27

    Parameters
    ----------
    lags : numpy.ndarray
        Space (x) and time (t) lag given as tuple (x, t) or an array of
        shape (n, 2) holding n lag pairs.
    model : Callable
        Theoretical variogram function from :mod:`skgstat.models` of
        signature ``model(h, r, c0, [s], b)``.
    rx : float
        Effective range along the space axis.
    rt : float
        Effective range along the time axis.
    args : float
        All further parameters of the model: the sill, the shape parameter
        for the stable and matern model and the nugget.

    Returns
    -------
    gamma : float, numpy.ndarray
        The semi-variance modeled for the given lags.

    Notes
    -----
    This model is implemented like:

    .. math::
        \gamma (h,t) = \gamma_m \left(\sqrt{h^2 + \left(\frac{r_x}{r_t}t\right)^2}\right)

    Where :math:`\gamma_m` is the theoretical model with the effective
    range :math:`r_x`. Thus, :math:`r_x` and :math:`r_t` are the effective
    ranges along the space and time axis. The metric model is described
    by Gräler et al. [17]_.

    References
    ----------
    .. [17] Gräler, B., Pebesma, E. and Heuvelink, G. (2016),
       Spatio-Temporal Interpolation using gstat. The R Journal 8(1),
       204-218.

    """
    h, t = np.asarray(lags, dtype=float).T
    return model(np.sqrt(h**2 + (rx / rt * t)**2), rx, *args)
