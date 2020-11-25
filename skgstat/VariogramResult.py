import numpy as np
import pandas as pd
from .Variogram import Variogram

class LagClass(object):
    def __init__(self, size):
        self.size = size

class VariogramResult(Variogram):
    """This class represents a calculated variogram without the original
    coordinates and values. This is then a much smaller object and can
    easily be stored in a database etc.

    Usage:

      v = skgstat.Variogram(coordinates, values)
      vr = VariogramResult(v) # Throw away coordinates and values (and intermediate large arrays)

      ...

      ok = skgstat.OrdinaryKriging(vr.with_values(coordinates, values), **kriging_args)
    """
    def __init__(self, variogram):
        info = variogram.describe()
        Variogram.__init__(self, model=info["name"], estimator=info["estimator"])
        self.cof = variogram.cof.copy()
        self._bins = variogram.bins.copy()
        self.__experimental = variogram.experimental.copy()
        self._lag_classes = [LagClass(cls.size) for cls in variogram.lag_classes()]
    @classmethod
    def new_from_params(cls, cof, bins=None, experimental=None, lag_classes=None, **kw):
        self = object.__new__(cls)
        Variogram.__init__(self, **kw)
        self.cof = np.array(cof)
        if bins is not None:
            self._bins = bins
        if experimental is not None:
            self.__experimental = experimental
        if lag_classes is not None:
            self._lag_classes = lag_classes
        return self
    def fit(self, *arg, **kw):
        pass
    @property
    def _experimental(self):
        return self.__experimental
    def set_values(self, values, **kw):
        if values is not None:
            Variogram.set_values(self, values, **kw)
    def _calc_distances(self, force=False):
        pass
    def _calc_diff(self, force=False):
        pass
    def _calc_groups(self, force=False):
        pass
    def lag_classes(self):
        return self._lag_classes
    def with_values(self, coordinates, values):
        v = VariogramResult(self)
        v._X = coordinates
        v._values = values
        return v
