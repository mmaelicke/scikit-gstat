from .Variogram import Variogram
from .DirectionalVariogram import DirectionalVariogram
from .SpaceTimeVariogram import SpaceTimeVariogram
from .Kriging import OrdinaryKriging
from .MetricSpace import MetricSpace, MetricSpacePair, ProbabalisticMetricSpace, RasterEquidistantMetricSpace
from . import interfaces
from . import data
from . import util

# set some stuff
from .__version__ import __version__
__author__ = 'Mirko Maelicke <mirko.maelicke@kit.edu>'
__backend__ = 'matplotlib'
