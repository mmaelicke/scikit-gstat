import inspect

import numpy as np
from scipy.optimize import minimize

import skgstat as skg
from skgstat import models
import skgstat.util.likelihood as li


def test_wrapped_model_doc():
    """Test the docstring wrapping"""
    # create a wrapped spherical function
    wrapped = li._model_transformed(models.spherical, has_s=False)

    assert 'Autocorrelation function.' in wrapped.__doc__
    assert models.spherical.__doc__ in wrapped.__doc__


def test_wrapped_model_args():
    """"
    Check that the number of model parameters is initialized correctly.
    """
    # get the two functions
    spherical = li._model_transformed(models.spherical, has_s=False)
    stable = li._model_transformed(models.stable, has_s=True)

    sig = inspect.signature(spherical)
    assert len(sig.parameters) == 2

    sig = inspect.signature(stable)
    assert len(sig.parameters) == 3


def test_build_A():
    """
    Test the autocorrelation matrix building.
    """
    # create a wrapped spherical function
    wrap_2 = li._model_transformed(models.spherical, has_s=False)
    wrap_3 = li._model_transformed(models.stable, has_s=True)

    # build the autocorrelation matrix
    A_5 = li._build_A(wrap_2, [1, 1, 0], np.arange(0, 1, 0.1))
    A_6 = li._build_A(wrap_3, [1, 1, 1, 1], np.arange(0, 1.5, 0.1))

    # check the matrix shape
    assert A_5.shape == (5, 5)
    assert A_6.shape == (6, 6)


def test_likelihood():
    """
    Call the likelihood function and make sure that it optimizes the
    the pancake variogram
    """
    # build the variogram from the tutorial
    c, v = skg.data.pancake(300, seed=42).get('sample')
    vario = skg.Variogram(c, v, bin_func='scott', maxlag=0.7)

    # get the likelihood function
    like = li.get_likelihood(vario)

    # cretae the optimization attributes
    sep_mean = vario.distance.mean()
    sam_var = vario.values.var()

    # create initial guess
    p0 = np.array([sep_mean, sam_var, 0.1 * sam_var])

    # create the bounds to restrict optimization
    bounds = [[0, vario.bins[-1]], [0, 3*sam_var], [0, 2.9*sam_var]]

    # minimize the likelihood function
    res = minimize(like, p0, bounds=bounds, method='SLSQP')

    # the result and p0 should be different
    assert not np.allclose(res.x, p0, rtol=1e-3)
