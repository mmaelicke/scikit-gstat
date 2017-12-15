"""
TODO

test Variogram_Wrapper and debug_spherical
"""
import unittest
from skgstat.models import spherical, exponential, gaussian, cubic, stable, matern


class TestModels(unittest.TestCase):
    def setUp(self):
        """
        Setting up the values

        :param h:   the separation lag
        :param a:   the range parameter (not effective range!)
        :param C0:  the Variogram sill
        :param b:   the Variogram nugget
        """

        self.h = 1
        self.a = 2
        self.c0 = 3
        self.b = 4
        self.s = 5

    def test_spherical(self):
        """
        Testing spherical model
        """

        self.assertAlmostEqual(spherical(self.h, self.a, self.c0, self.b), 3.3125)

    def test_exponential(self):
        """
        Testing exponential model
        """

        self.assertAlmostEqual(exponential(self.h, self.a, self.c0, self.b), 6.330609519554711)

    def test_gaussian(self):
        """
        Testing gaussian model
        """

        self.assertAlmostEqual(gaussian(self.h, self.a, self.c0, self.b), 3.472366552741015)

    def test_cubic(self):
        """
        Testing cubic model
        """

        self.assertAlmostEqual(cubic(self.h, self.a, self.c0, self.b), 3.240234375)

    def test_stable(self):
        """
        Testing stable model
        """

        self.assertAlmostEqual(stable(self.h, self.a, self.c0, self.s, self.b), 3.8512082197805455)

    def test_matern(self):
        """
        Testing matern model
        """

        self.assertAlmostEqual(matern(self.h, self.a, self.c0, self.s, self.b), 3.984536090177115)

