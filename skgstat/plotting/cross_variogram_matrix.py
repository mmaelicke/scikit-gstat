from typing import List
import matplotlib as plt
from skgstat.Variogram import Variogram

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
except ImportError:
    pass


def matplotlib_cv_matrix(
    variograms: List[List[Variogram]],
    add_model: bool = True,
    sharex: bool = True,
    sharey: bool = True
):
    """"""
    raise NotImplementedError


def plotly_cv_matrix(
    variograms: List[List[Variogram]],
    add_model: bool = True,
    sharex: bool = True,
    sharey: bool = True
):
    """"""
    raise NotImplementedError
