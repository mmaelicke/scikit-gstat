from .Variogram import Variogram
from .DirectionalVariogram import DirectionalVariogram
from .SpaceTimeVariogram import SpaceTimeVariogram
from .Kriging import OrdinaryKriging
from .MetricSpace import MetricSpace, MetricSpacePair
from . import interfaces

# set some stuff
__version__ = '0.5.1'
__author__ = 'Mirko Maelicke <mirko.maelicke@kit.edu>'
__backend__ = 'matplotlib'
