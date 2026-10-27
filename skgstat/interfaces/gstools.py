"""GSTools Interface."""
import numpy as np
import warnings


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
    Instantiate a corresponding GSTools :any:`CovModel <gstools.CovModel>`.

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

    Warns
    -----
    Warning
        If the Variogram is a cross-variogram


    Returns
    -------
    model : :any:`CovModel <gstools.CovModel>`
        Corresponding GSTools covmodel.

    Note
    ----
    In case you intend to use the
    :func:`coordinates <skgstat.Variogram.coordinates>`
    in a GSTools workflow, you need to transpose the coordinate
    array like:

    >> cond_pos Variogram.coordinates.T

    """
    # try to import gstools and notify user if not installed
    try:
        import gstools as gs
    except ImportError as e:  # pragma: no cover
        raise ImportError("to_gstools: GSTools not installed.") from e

    # at least gstools>=1.3.0 is needed
    if list(map(int, gs.__version__.split(".")[:2])) < [1, 3]:  # pragma: no cover
        raise ValueError("to_gstools: GSTools v1.3 or greater required.")

    # if Variogram is a cross-variogram warn the user
    if variogram.is_cross_variogram:
        warnings.warn("This instance is a cross-variogram!!" +
            " GSTools.CovModel will most likely not handle this Variogram correctly.")


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
    # the skgstat sill is already the partial sill
    gs_kwargs = dict(
        var=float(describe["sill"]),
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
    Instantiate a GSTools Krige class.

    This can only export isotropic models.
    Note: the `fit_variogram` is always set to `False`

    Parameters
    ----------
    variogram : skgstat.Variogram
        Scikit-GStat Variogram instamce
    **kwargs
        Keyword arguments forwarded to GSTools Krige.
        Refer to :any:`Krige <gstools.krige.Krige>` to
        learn about all possible options.
        Note that the `fit_variogram` parameter will
        always be False.

    Raises
    ------
    ImportError
        When GSTools is not installed.
    ValueError
        When GSTools version is not v1.3 or greater.
    ValueError
        When given Variogram model is not supported ('harmonize').

    Warns
    -----
    Warning
        If the Variogram is a cross-variogram

    Returns
    -------
    :any:`Krige <gstools.krige.Krige>`
        Instantiated GSTools Krige class.

    See Also
    --------
    gstools.Krige

    """
    # try to import gstools and notify user if not installed
    try:
        import gstools as gs
    except ImportError as e:  # pragma: no cover
        raise ImportError("to_gstools: GSTools not installed.") from e

    # at least gstools>=1.3.0 is needed
    if list(map(int, gs.__version__.split(".")[:2])) < [1, 3]:  # pragma: no cover
        raise ValueError("to_gstools: GSTools v1.3 or greater required.")

    # convert variogram to a CovModel
    model = skgstat_to_gstools(variogram=variogram)

    # extract cond_pos and cond_vals
    cond_pos = variogram.coordinates.T
    cond_vals = variogram.values

    # disable the re-fitting of the variogram in gstools
    kwargs['fit_variogram'] = False

    # instantiate the Krige class
    krige = gs.krige.Krige(model, cond_pos, cond_vals, **kwargs)

    # return the class
    return krige


def stvariogram_to_gstools(stvariogram, **kwargs):
    """
    Instantiate a spatio-temporal GSTools :any:`CovModel <gstools.CovModel>`.

    .. versionadded:: 1.0.27

    Only the metric space-time model can be exported. The CovModel is
    temporal, the time is the last dimension and the effective ranges of
    space and time become the length scales.

    Parameters
    ----------
    stvariogram : :any:`SpatioTemporalVariogram`
        Scikit-Gstat SpatioTemporalVariogram instance.
    **kwargs
        Keyword arguments forwarded to the instantiated GSTools CovModel.
        The default parameters 'temporal', 'spatial_dim', 'var',
        'len_scale', 'nugget', 'rescale' and optional shape parameters
        will be extracted from the given SpatioTemporalVariogram but they
        can be overwritten here.

    Raises
    ------
    ImportError
        When GSTools is not installed.
    ValueError
        When GSTools version is not v1.5 or greater.
    ValueError
        When the space-time model is not 'metric'.

    Returns
    -------
    model : :any:`CovModel <gstools.CovModel>`
        Corresponding spatio-temporal GSTools covmodel.

    Note
    ----
    The length scale of the time axis is given in the internal time unit
    of the SpatioTemporalVariogram, which is seconds for datetime-like
    input.

    """
    # try to import gstools and notify user if not installed
    try:
        import gstools as gs
    except ImportError as e:  # pragma: no cover
        raise ImportError("to_gstools: GSTools not installed.") from e

    # at least gstools>=1.5.0 is needed for spatio-temporal models
    if list(map(int, gs.__version__.split(".")[:2])) < [1, 5]:  # pragma: no cover
        raise ValueError("to_gstools: GSTools v1.5 or greater required.")

    # gstools only supports metric space-time models
    if stvariogram._model_name != 'metric':
        raise ValueError(
            "to_gstools: GSTools only supports metric space-time models, "
            "but the model is '%s'. Use model='metric'." % stvariogram._model_name
        )

    # extract all needed settings
    params = stvariogram.parameters
    name = stvariogram._marginal_model_name
    gs_describe = MODEL_MAP[name]

    # the time is appended as last dimension with its own length scale
    spatial_dim = stvariogram.coordinates.shape[1]
    gs_kwargs = dict(
        temporal=True,
        spatial_dim=spatial_dim,
        var=float(params["sill"]),
        len_scale=[float(params["space_range"])] * spatial_dim + [float(params["time_range"])],
        nugget=float(params["nugget"]),
    )

    # some skgstat models need different rescale
    rescale = gs_describe.get("rescale", 1.0)
    gs_kwargs["rescale"] = rescale(params) if callable(rescale) else rescale
    arg_map = gs_describe.get("arg_map", dict())
    for arg in arg_map:
        gs_kwargs[arg] = float(params[arg_map[arg]])

    # update the parameters
    gs_kwargs.update(kwargs)

    # get the model and return the CovModel
    gs_model = getattr(gs, gs_describe["gs_cls"])
    return gs_model(**gs_kwargs)


def stvariogram_to_krige(stvariogram, **kwargs):
    """
    Instantiate a spatio-temporal GSTools Krige class.

    .. versionadded:: 1.0.27

    Only the metric space-time model can be exported.
    Note: the `fit_variogram` is always set to `False`

    Parameters
    ----------
    stvariogram : skgstat.SpatioTemporalVariogram
        Scikit-GStat SpatioTemporalVariogram instance
    **kwargs
        Keyword arguments forwarded to GSTools Krige.
        Refer to :any:`Krige <gstools.krige.Krige>` to
        learn about all possible options.
        Note that the `fit_variogram` parameter will
        always be False.

    Raises
    ------
    ImportError
        When GSTools is not installed.
    ValueError
        When GSTools version is not v1.5 or greater.
    ValueError
        When the space-time model is not 'metric'.

    Returns
    -------
    :any:`Krige <gstools.krige.Krige>`
        Instantiated GSTools Krige class.

    See Also
    --------
    gstools.Krige

    """
    # try to import gstools and notify user if not installed
    try:
        import gstools as gs
    except ImportError as e:  # pragma: no cover
        raise ImportError("to_gstools: GSTools not installed.") from e

    # at least gstools>=1.5.0 is needed for spatio-temporal models
    if list(map(int, gs.__version__.split(".")[:2])) < [1, 5]:  # pragma: no cover
        raise ValueError("to_gstools: GSTools v1.5 or greater required.")

    # convert variogram to a CovModel
    model = stvariogram_to_gstools(stvariogram=stvariogram)

    # extract cond_pos and cond_vals, the time is the last dimension
    cond_pos = np.column_stack((stvariogram.coordinates, stvariogram.time)).T
    cond_vals = stvariogram.values

    # disable the re-fitting of the variogram in gstools
    kwargs['fit_variogram'] = False

    # instantiate the Krige class
    krige = gs.krige.Krige(model, cond_pos, cond_vals, **kwargs)

    # return the class
    return krige
