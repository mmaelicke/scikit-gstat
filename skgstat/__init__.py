from .Variogram import Variogram
from .DirectionalVariogram import DirectionalVariogram
from .SpaceTimeVariogram import SpaceTimeVariogram
from .Kriging import OrdinaryKriging
from . import interfaces

# set some stuff
__version__ = '0.3.8'
__author__ = 'Mirko Maelicke <mirko.maelicke@kit.edu>'
__backend__ = 'matplotlib'
