import pytest

from skgstat import data
import numpy as np
from numpy.testing import assert_array_almost_equal


def test_data_names():
    assert 'pancake' in data.names


def test_loader():
    img = data.pancake_field().get('sample')

    assert img.shape[0] == 500 and img.shape[1] == 500


def test_sample():
    c, v = data.pancake(N=50).get('sample')

    assert len(c) == len(v) == 50


def test_loader_mean():
    c0, v0 = data.pancake(N=10, band=0).get('sample')
    c1, v1 = data.pancake(N=10, band=1).get('sample')
    c2, v2 = data.pancake(N=10, band=2).get('sample')
    cm, cv = data.pancake(N=10, band='mean').get('sample')

    # manually calculate the mean
    mean = np.mean(np.column_stack((v0, v1, v2)), axis=1)
    print(mean)

    assert_array_almost_equal(cv, mean, decimal=4)


def test_aniso_data():
    assert 'aniso' in data.names

    img = data.aniso_field().get('sample')
    assert img.shape[0] == 500 and img.shape[1] == 500

    c, v = data.aniso(N=25).get('sample')
    assert len(c) == len(v) == 25


def test_meuse_loads():
    df = data._loader.read_sample_file('meuse.txt')

    # get zinc data
    _, zinc = data.meuse(variable='zinc').get('sample')

    assert_array_almost_equal(
        zinc, df[['zinc']].values, decimal=6
    )

    # check exeption
    with pytest.raises(AttributeError) as e:
        data.meuse(variable='unknown')

    assert 'variable has to be in' in str(e.value)
