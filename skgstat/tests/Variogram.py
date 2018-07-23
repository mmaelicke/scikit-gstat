import unittest

import numpy as np

from skgstat import Variogram

class TestVariogram(unittest.TestCase):
    def test_standard_settings(self):
        # check the parameters
        np.random.seed(42)
        c = np.random.gamma(10, 4, (30, 2))
        np.random.seed(42)
        v = np.random.normal(10, 4, 30)

        V = Variogram(c, v)

        for x,y in zip(V.parameters, [439.405, 281.969, 0]):
            self.assertAlmostEqual(x, y, places=3)


if __name__ == '__main__':
    unittest.main()
