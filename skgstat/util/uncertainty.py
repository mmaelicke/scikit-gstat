"""
Estimate uncertainties propagated through the Variogram
using a MonteCarlo approach
"""
from typing import Union, List
from skgstat import Variogram
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


def propagate(
    variogram: Variogram = None,
    source: Union[str, List[str]] = 'values',
    sigma: Union[float, List[float]] = 5,
    evalf: Union[str, List[str]] = 'experimental',
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
    evalf : list
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
        Defaults to ``500``.
    eval_at : int
        If evalf is set to model, the theoretical model get evaluated
        at this many evenly spaced lags up to maximum lag.
        Defaults to ``100``.
    n_jobs : int
        The evaluation can be performed in parallel. This will specify
        how many processes may be spawned in parallel. None will spwan
        only one (default).

        .. note::
            This is an untested experimental feature.

    Returns
    -------
    conf_interval : numpy.ndarray
        Confidence interval of the uncertainty propagation as
        [lower, median, upper]. If more than one evalf is given, a
        list of ndarrays will be returned.
        See notes for more details.

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
    can be changed using the eval_at parameter.

    If more than one evalf parameter is given, the Variogram will be
    evaluated at multiple steps and each one will be returned as a
    confidence interval. Thus if ``len(evalf) == 2``, a list containing
    two confidence interval matrices will be returned.
    The order is [experimental, parameter, model].

    """
    # handle error bounds shortcut
    if use_bounds:
        kwargs['q'] = 0

    # extract the MetricSpace to speed things up a bit
    metricSpace = variogram._X

    # get the source of error
    if isinstance(source, str):
        source = [source]

    if not isinstance(sigma, (list, tuple)):
        sigma = [sigma]

    if isinstance(evalf, str):
        evalf = [evalf]

    # get the static variogram parameters
    _var_opts = variogram.describe().get('params', {})
    omit_names = [*source, 'verbose']
    args = {k: v for k, v in _var_opts.items() if k not in omit_names}

    # add back the metric space
    args['coordinates'] = metricSpace

    # build the parameter field
    num_iter = kwargs.get('num_iter', 500)
    rng = np.random.default_rng(kwargs.get('seed'))
    dist = getattr(rng, kwargs.get('distribution', 'normal'))
    param_field = []

    for it in range(num_iter):
        par = {**args}

        # add the noisy params
        for s, err in zip(source, sigma):
            obs = getattr(variogram, s)
            size = len(obs) if hasattr(obs, '__len__') else 1
            par[s] = dist(obs, err, size=size)

        # append to param field
        param_field.append(par)

    # define the eval function
    def func(par):
        vario = Variogram(**par)
        out = []
        if 'experimental' in evalf:
            out.append(vario.experimental)
        if 'parameter' in evalf:
            out.append(vario.parameters)
        if 'model' in evalf:
            x = np.linspace(0, np.max(vario.bins), kwargs.get('eval_at', 100))
            out.append(vario.fitted_model(x))
        return out

    # build the worker
    worker = Parallel(n_jobs=kwargs.get('n_jobs'))
    if verbose:
        generator = (delayed(func)(par) for par in tqdm(param_field))
    else:
        generator = (delayed(func)(par) for par in param_field)

    # run
    result = worker(generator)

    # split up conf intervals
    conf_intervals = []

    for i in range(len(evalf)):
        # unpack
        res = [result[j][i] for j in range(len(result))]

        # create the result
        ql = int(kwargs.get('q', 10) / 2)
        qu = 100 - int(kwargs.get('q', 10) / 2)
        conf_intervals.append(
            np.column_stack((
                np.percentile(res, ql, axis=0),
                np.median(res, axis=0),
                np.percentile(res, qu, axis=0)
            ))
        )

    # return
    if len(conf_intervals) == 1:
        return conf_intervals[0]
    return conf_intervals
