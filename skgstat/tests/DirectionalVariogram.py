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

    def test_azimuth(self):
        DV = DirectionalVariogram(self.c, self.v, azimuth=-45)

        for x, y in zip(DV.parameters, [469.571, 195.996, 0]):
            self.assertAlmostEqual(x, y, places=3)

    def test_invalid_azimuth(self):
        with self.assertRaises(ValueError) as e:
            DV = DirectionalVariogram(self.c, self.v, azimuth=360)
            self.assertEqual(
                str(e),
                'The azimuth is an angle in degree and has '
                'to meet -180 <= angle <= 180'
            )

    def test_tolerance(self):
        DV = DirectionalVariogram(self.c, self.v, tolerance=15)

        for x, y in zip(DV.parameters, [964.45, 953.375, 0]):
            self.assertAlmostEqual(x, y, places=3)

    def test_invalid_tolerance(self):
        with self.assertRaises(ValueError) as e:
            DV = DirectionalVariogram(self.c, self.v, tolerance=-1)
            self.assertEqual(
                str(e),
                'The tolerance is an angle in degree and has to '
                'meet 0 <= angle <= 360'
            )


if __name__ == '__main__':
    unittest.main()
