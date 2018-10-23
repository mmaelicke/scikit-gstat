import unittest

import numpy as np

from skgstat import DirectionalVariogram


class TestDirectionalVariogramInstantiation(unittest.TestCase):
    def setUp(self):
        # set up default valures, whenever c,v are not important
        np.random.seed(42)
        self.c = np.random.gamma(10, 4, (30, 2))
        np.random.seed(42)
        self.v = np.random.normal(10, 4, 30)

    def test_standard_settings(self):
        DV = DirectionalVariogram(self.c, self.v)

        for x, y in zip(DV.parameters, [1046.178, 564.075, 0]):
            self.assertAlmostEqual(x, y, places=3)


if __name__ == '__main__':
    unittest.main()
