"""
Estimate uncertainties propagated through the Variogram
using a MonteCarlo approach
"""
from typing import Union, List
from uncertainty_framework import MonteCarlo
from skgstat import Variogram
import numpy as np


def _propagate_experimental(**kwargs):
    vario = Variogram(**kwargs)

    return np.asarray(vario.experimental)


def _propagate_params(**kwargs):
    vario = Variogram(**kwargs)

    return vario.parameters


def _propagate_model(eval_at=100, **kwargs):
    vario = Variogram(**kwargs)

    x = np.linspace(0, np.max(vario.bins), num=eval_at)

    return vario.fitted_model(x)


def propagate(
    variogram: Variogram = None,
    source: Union[str, List[str]] = 'values',
    sigma: Union[float, List[float]] = 5,
    evalf: str = 'experimental',
    verbose: bool = False,
    use_bounds: bool = False,
    **kwargs
):
    """
    Uncertainty propagation for the variogram.
    For a given :class:`Variogram <skgstat.Variogram>`
    instance a source of error and scale of error
    distribution can be specified. The function will
    propagate the uncertainty into different parts of
    the :class:`Variogram <skgstat.Variogram>` and
    return the confidence intervals or error bounds.

    Parameters
    ----------
    variogram : skgstat.Variogram
        The base variogram. The variogram parameters will
        be used as fixed arguments for the Monte Carlo
        simulation.
    source : list
        Source of uncertainty. This has to be an attribute
        of :class:`Variogram <skgstat.Variogram>`. Right
        now only ``'values'`` is really supported, anything
        else is untested.
    sigma : list
        Standard deviation of the error distribution.
    evalf : str
        Evaluation function. This specifies, which part of
        the :class:`Variogram <skgstat.Variogram>` should be
        used to be evaluated. Possible values are
        ``'experimental'`` for the experimental variogram,
        ``'model'`` for the fitted model and ``parameter'``
        for the variogram parameters
    verbose : bool
        If True, the uncertainty_framework package used under
        the hood will print a progress bar to the console.
        Defaults to False.
    use_bounds : bool
        Shortcut to set the confidence interval bounds to the
        minimum and maximum value and thus return the error
        margins over a confidence interval.

    Keyword Arguments
    -----------------
    distribution : str
        Any valid :any:`numpy.random` distribution function, that
        takes the scale as argument.
        Defaults to ``'normal'``.
    q : int
        Width (percentile) of the confidence interval. Has to be a
        number between 0 and 100. 0 will result in the minimum and
        maximum value as bounds. 100 turns both bounds into the
        median value.
        Defaults to ``10``
    num_iter : int
        Number of iterations used in the Monte Carlo simulation.
        Defaults to ``5000``.
    eval_at : int
        If evalf is set to model, the theoretical model get evaluated
        at this many evenly spaced lags up to maximum lag.
        Defaults to ``100``.

    Returns
    -------
    conf_interval : numpy.ndarray
        Confidence interval of the uncertainty propagation as
        [lower, median, upper]. See notes for more details

    Notes
    -----
    For each member of the evaluated property, the lower and upper bound
    along with the median value is retuned as ``[low, median, up]``.
    Thus the returned array has the shape ``(N, 3)``.
    N is the lengh of evaluated property, which is
    :func:`n_lags <skgstat.Variogram.n_lags` for ``'experimental'``,
    either ``3`` for ``'parameter'`` or ``4`` if
    :func:`Variogram.model = 'stable' | 'matern' <skgstat.Variogram.model>`
    and ``100`` for ``'model'`` as the model gets evaluated at
    100 evenly spaced lags up to the maximum lag class. This amount
    can be changed using the eval_at parameter

    """
    # handle error bounds shortcut
    if use_bounds:
        kwargs['q'] = 0

    # extract the MetricSpace to speed things a bit up
    metricSpace = variogram._X

    # get the source of error
    if isinstance(source, str):
        source = [source]

    if not isinstance(sigma, (list, tuple)):
        sigma = [sigma]

    # get the variogram parameters
    _var_opts = variogram.describe().get('params', {})
    omit_names = [*source, 'verbose']
    args = {k: v for k, v in _var_opts.items() if k not in omit_names}

    # add the metric space
    args['coordinates'] = metricSpace

    # build the parameter map
    parameters = dict()
    for s, err in zip(source, sigma):
        obs = getattr(variogram, s)
        parameters[s] = dict(
            distribution=kwargs.get('distribution', 'normal'),
            scale=err,
            value=obs
        )

    # switch the evaluation function
    if evalf == 'experimental':
        func = _propagate_experimental
    elif evalf == 'parameter':
        func = _propagate_params
    elif evalf == 'model':
        func = _propagate_model
    else:
        raise AttributeError('')

    # build the montecarlo object
    mc = MonteCarlo(
        func=func,
        num_iter=kwargs.get('num_iter', 500),
        parameters=parameters,
        verbose=verbose,
        **args
    )

    # run
    res = mc.run()

    # create the result
    ql = int(kwargs.get('q', 10) / 2)
    qu = 100 - int(kwargs.get('q', 10) / 2)
    conf_interval = np.column_stack((
        np.percentile(res, ql, axis=0),
        np.median(res, axis=0),
        np.percentile(res, qu, axis=0)
    ))

    return conf_interval
