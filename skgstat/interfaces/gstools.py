"""GSTools Interface."""
import numpy as np


def stable_rescale(describe):
    """Get GSTools rescale parameter from sk-gstat stable model description."""
    return np.power(3, 1 / describe["shape"])


def matern_rescale(describe):
    """Get GSTools rescale parameter from sk-gstat matern model description."""
    if 0.5 < describe["smoothness"] < 10.0:
        return 4.
    return 6.


MODEL_MAP = dict(
    spherical=dict(gs_cls="Spherical"),
    exponential=dict(gs_cls="Exponential", rescale=3.0),
    gaussian=dict(gs_cls="Gaussian", rescale=2.0),
    cubic=dict(gs_cls="Cubic"),
    stable=dict(
        gs_cls="Stable", arg_map={"alpha": "shape"}, rescale=stable_rescale
    ),
    matern=dict(
        gs_cls="Matern", arg_map={"nu": "smoothness"}, rescale=matern_rescale
    ),
)


def skgstat_to_gstools(variogram, **kwargs):
    """Instantiate a corresponding GSTools CovModel."""
    try:
        import gstools as gs
    except ImportError:
        raise ImportError("skgstat_to_gstools: GSTools needs to be installed.")
    kwargs.setdefault("dim", variogram.coordinates.ndim)
    describe = variogram.describe()
    name = describe["name"]
    if name not in MODEL_MAP:
        raise ValueError("skgstat_to_gstools: model not supported: " + name)
    gs_describe = MODEL_MAP[name]
    gs_describe.setdefault("rescale", 1.0)
    gs_describe.setdefault("arg_map", dict())
    gs_kwargs = dict(
        var=float(describe["sill"] - describe["nugget"]),
        len_scale=float(describe["effective_range"]),
        nugget=float(describe["nugget"]),
    )
    rescale = gs_describe["rescale"]
    gs_kwargs["rescale"] = rescale(describe) if callable(rescale) else rescale
    for arg in gs_describe["arg_map"]:
        gs_kwargs[arg] = float(describe[gs_describe["arg_map"][arg]])
    gs_kwargs.update(kwargs)
    gs_model = getattr(gs, MODEL_MAP[name]["gs_cls"])
    return gs_model(**gs_kwargs)
