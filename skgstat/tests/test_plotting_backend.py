import pytest
from skgstat.plotting import backend

import matplotlib.pyplot as plt
import plotly.graph_objects as go 


def test_backend_no_args():
    """
    The default backend should be 'matplotlib'
    """
    assert backend() == 'matplotlib'


@pytest.mark.depends(on=['test_backend_no_args'])
def test_raise_value_error():
    """
    Raise a value error by setting the wrong backend
    """
    with pytest.raises(ValueError):
        backend('not-a-backend')


@pytest.mark.depends(on=['test_raise_value_error'])
def test_change_plotting_backend():
    """
    Set the correct backend and check
    """
    # change to plotly
    backend('plotly')
    assert backend() == 'plotly'

    # change back
    backend('matplotlib')
