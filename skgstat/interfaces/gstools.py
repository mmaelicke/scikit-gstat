"""GSTools Interface."""
import numpy as np


def stable_rescale(describe):
    """Get GSTools rescale parameter from sk-gstat stable model description."""
    return np.power(3, 1 / describe["shape"])


MODEL_MAP = dict(
    spherical=dict(gs_cls="Spherical"),
    exponential=dict(gs_cls="Exponential", rescale=3.0),
    gaussian=dict(gs_cls="Gaussian", rescale=2.0),
    cubic=dict(gs_cls="Cubic"),
    stable=dict(
        gs_cls="Stable", arg_map={"alpha": "shape"}, rescale=stable_rescale
    ),
    matern=dict(
        gs_cls="Matern", arg_map={"nu": "smoothness"}, rescale=4.0
    ),
)


def skgstat_to_gstools(variogram, **kwargs):
    """
    Instantiate a corresponding GSTools CovModel.

    By default, this will be an isotropic model.

    Parameters
    ----------
    variogram : :any:`Variogram`
        Scikit-Gstat Variogram instance.
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

    Returns
    -------
    :any:`CovModel`
        Corresponding GSTools covmodel.
    """
    # try to import gstools and notify user if not installed
    try:
        import gstools as gs
    except ImportError as e:  # pragma: no cover
        raise ImportError("to_gstools: GSTools not installed.") from e

    # at least gstools>=1.3.0 is needed
    if list(map(int, gs.__version__.split(".")[:2])) < [1, 3]:  # pragma: no cover
        raise ValueError("to_gstools: GSTools v1.3 or greater requiered.")

    # gstolls needs the spatial dimension
    kwargs.setdefault("dim", variogram.dim)

    # extract all needed settings
    describe = variogram.describe()

    # get the theoretical model name
#    name = describe["name"]
    name = describe['model']

    if name not in MODEL_MAP:
        raise ValueError("skgstat_to_gstools: model not supported: " + name)
    gs_describe = MODEL_MAP[name]

    # set variogram parameters
    gs_describe.setdefault("rescale", 1.0)
    gs_describe.setdefault("arg_map", dict())
    gs_kwargs = dict(
        var=float(describe["sill"] - describe["nugget"]),
        len_scale=float(describe["effective_range"]),
        nugget=float(describe["nugget"]),
    )

    # some skgstat models need different rescale
    rescale = gs_describe["rescale"]
    gs_kwargs["rescale"] = rescale(describe) if callable(rescale) else rescale
    arg_map = gs_describe["arg_map"]
    for arg in arg_map:
        gs_kwargs[arg] = float(describe[arg_map[arg]])

    # update the parameters
    gs_kwargs.update(kwargs)

    # get the model and return the CovModel
    gs_model = getattr(gs, gs_describe["gs_cls"])
    return gs_model(**gs_kwargs)


def skgstat_to_krige(variogram, **kwargs):
    """
    Instatiate a GSTools Krige class.

    This can only export isotropic models.
    Note: the `fit_variogram` is always set to `False`

    Parameters
    ----------
    variogram : skgstat.Variogram
        Scikit-GStat Variogram instamce
    **kwargs
        Keyword arguments forwarded to GSTools Krige.

    Keyword Arguments
    -----------------
    drift_functions : :class:`list` of :any:`callable`, :class:`str` or :class:`int`
        Either a list of callable functions, an integer representing
        the polynomial order of the drift or one of the following strings:

            * "linear" : regional linear drift (equals order=1)
            * "quadratic" : regional quadratic drift (equals order=2)

    ext_drift : :class:`numpy.ndarray` or :any:`None`, optional
        the external drift values at the given cond. positions.
    mean : :class:`float`, optional
        mean value used to shift normalized conditioning data.
        Could also be a callable. The default is None.
    normalizer : :any:`None` or :any:`Normalizer`, optional
        Normalizer to be applied to the input data to gain normality.
        The default is None.
    trend : :any:`None` or :class:`float` or :any:`callable`, optional
        A callable trend function. Should have the signiture: f(x, [y, z, ...])
        This is used for detrended kriging, where the trended is subtracted
        from the conditions before kriging is applied.
        This can be used for regression kriging, where the trend function
        is determined by an external regression algorithm.
        If no normalizer is applied, this behaves equal to 'mean'.
        The default is None.
    unbiased : :class:`bool`, optional
        Whether the kriging weights should sum up to 1, so the estimator
        is unbiased. If unbiased is `False` and no drifts are given,
        this results in simple kriging.
        Default: True
    exact : :class:`bool`, optional
        Whether the interpolator should reproduce the exact input values.
        If `False`, `cond_err` is interpreted as measurement error
        at the conditioning points and the result will be more smooth.
        Default: False
    cond_err : :class:`str`, :class :class:`float` or :class:`list`, optional
        The measurement error at the conditioning points.
        Either "nugget" to apply the model-nugget, a single value applied to
        all points or an array with individual values for each point.
        The "exact=True" variant only works with "cond_err='nugget'".
        Default: "nugget"
    pseudo_inv : :class:`bool`, optional
        Whether the kriging system is solved with the pseudo inverted
        kriging matrix. If `True`, this leads to more numerical stability
        and redundant points are averaged. But it can take more time.
        Default: True
    pseudo_inv_type : :class:`str` or :any:`callable`, optional
        Here you can select the algorithm to compute the pseudo-inverse matrix:

            * `"pinv"`: use `pinv` from `scipy` which uses `lstsq`
            * `"pinv2"`: use `pinv2` from `scipy` which uses `SVD`
            * `"pinvh"`: use `pinvh` from `scipy` which uses eigen-values

        If you want to use another routine to invert the kriging matrix,
        you can pass a callable which takes a matrix and returns the inverse.
        Default: `"pinv"`
    fit_normalizer : :class:`bool`, optional
        Wheater to fit the data-normalizer to the given conditioning data.
        Default: False

    Raises
    ------
    ImportError
        When GSTools is not installed.
    ValueError
        When GSTools version is not v1.3 or greater.
    ValueError
        When given Variogram model is not supported ('harmonize').

    Returns
    -------
    :any:`Krige`
        Instantiated GSTools Krige class.

    Note
    ----
    The documentation for the keyword Arguments is directly
    taken from gstools==1.3.0 documentation. If you are running
    a more recent version, the arguments might differ.

    """
    # try to import gstools and notify user if not installed
    try:
        import gstools as gs
    except ImportError as e:  # pragma: no cover
        raise ImportError("to_gstools: GSTools not installed.") from e

    # at least gstools>=1.3.0 is needed
    if list(map(int, gs.__version__.split(".")[:2])) < [1, 3]:  # pragma: no cover
        raise ValueError("to_gstools: GSTools v1.3 or greater requiered.")

    # convert variogram to a CovModel
    model = skgstat_to_gstools(variogram=variogram)

    # extract cond_pos and cond_vals
    cond_pos = list(zip(*variogram.coordinates))
    cond_vals = variogram.values

    # disable the re-fitting of the variogram in gstools
    kwargs['fit_variogram'] = False

    # instantiate the Krige class
    krige = gs.krige.Krige(model, cond_pos, cond_vals, **kwargs)

    # return the class
    return krige
