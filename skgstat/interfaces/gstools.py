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
        raise ValueError("to_gstools: GSTools v1.3 or greater requiered.")

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
