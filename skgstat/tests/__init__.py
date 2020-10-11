"""
from skgstat.tests.estimator import TestEstimator
from skgstat.tests.models import TestModels, TestVariogramDecorator
from skgstat.tests.binning import TestEvenWidth, TestUniformCount
from skgstat.tests.Variogram import (
    TestVariogramInstatiation,
    TestVariogramArguments,
    TestVariogramFittingProcedure,
    TestVariogramQaulityMeasures,
    TestVariogramMethods,
    TestVariogramPlots,
)
from skgstat.tests.DirectionalVariogram import (
    TestDirectionalVariogramInstantiation,
    TestDirectionalVariogramMethods,
)
from skgstat.tests.SpaceTimeVariogram import (
    TestSpaceTimeVariogramInitialization,
    TestSpaceTimeVariogramArgumets,
    TestSpaceTimeVariogramPlots,
)
from skgstat.tests.kriging import (
    TestKrigingInstantiation,
    TestPerformance,
)
from skgstat.tests.interfaces import (
    TestVariogramEstimator,
    TestPyKrigeInterface,
    TestGstoolsInterface
)
from skgstat.tests.stmodels import (
    TestSumModel,
    TestProductModel,
    TestProductSumModel
)

import os
os.environ['SKG_SUPRESS'] = 'TRUE'
"""