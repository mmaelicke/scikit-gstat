"""
PyUnit Tests for the geostat.binning functions

TODO
Save arrays in another file?
Test direction and _in_bounds
"""

import unittest
import sys
from contextlib import contextmanager
from io import StringIO

import numpy as np
from numpy.testing import assert_array_almost_equal
from skgstat.binning import binify_even_width, binify_even_bin, group_to_bin, direction, _in_bounds

# TODO: test only against the first (or any other significant) line of results
# Result arrays
result_binify_even_width_bm = np.array(
    [0., 3., 6., 9., 0., 3., 6., 9., 0., 3., 6., 9., 0.,
     3., 6., 9., 0., 3., 6., 9.])

result_binify_even_width_bw = np.array(
    [0.42426407, 0.42426407, 0.42426407, 0.42426407, 0.42426407,
     0.42426407, 0.42426407, 0.42426407, 0.42426407, 0.42426407])

result_binify_even_width_bm_n6 = np.array(
    [0., 2., 4., 5., 0., 2., 4., 5., 0., 2., 4., 5., 0.,
     2., 4., 5., 0., 2., 4., 5.])

result_binify_even_width_bw_n6 = np.array(
    [0.70710678, 0.70710678, 0.70710678, 0.70710678, 0.70710678,
     0.70710678])

binify_dm = np.array([0., 1.41421356, 2.82842712, 4.24264069, 0.,
                       1.41421356, 2.82842712, 4.24264069, 0., 1.41421356,
                       2.82842712, 4.24264069, 0., 1.41421356, 2.82842712,
                       4.24264069, 0., 1.41421356, 2.82842712, 4.24264069])

result_binify_even_width_bm_maxlag = np.array([0., 3., 7., 9., 0., 3., 7., 9., 0., 3., 7., 9., 0.,
                                                3., 7., 9., 0., 3., 7., 9.])

result_binify_even_width_bw_maxlag = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])

result_binify_even_bin_bm = np.array([0., 2., 6., 8., 0., 2., 6., 8., 0., 2., 6., 8., 0.,
                                       2., 6., 8., 0., 2., 6., 8.])

result_binify_even_bin_bw = np.array([0., 0., 1.41421356, 0., 0.,
                                      0., 1.41421356, 0., 1.41421356, 0.])

result_binify_even_bin_bm_n6 = np.array([0., 1., 3., 5., 0., 1., 3., 5., 0., 1., 3., 5., 0.,
                                          1., 3., 5., 0., 1., 3., 5.])

result_binify_even_bin_bw_n6 = np.array([0., 1.41421356, 0., 1.41421356, 0.,
                                         1.41421356])

result_binify_even_bin_bm_maxlag = np.array([[0., 2., 6., 8., 0., 2., 6., 8., 0., 2., 6., 8., 0.,
                                              2., 6., 8., 0., 2., 6., 8.],
                                             [2., 0., 2., 6., 2., 0., 2., 6., 2., 0., 2., 6., 2.,
                                              0., 2., 6., 2., 0., 2., 6.],
                                             [6., 2., 0., 2., 6., 2., 0., 2., 6., 2., 0., 2., 6.,
                                              2., 0., 2., 6., 2., 0., 2.],
                                             [8., 6., 2., 0., 8., 6., 2., 0., 8., 6., 2., 0., 8.,
                                              6., 2., 0., 8., 6., 2., 0.],
                                             [0., 2., 6., 8., 0., 2., 6., 8., 0., 2., 6., 8., 0.,
                                              2., 6., 8., 0., 2., 6., 8.],
                                             [2., 0., 2., 6., 2., 0., 2., 6., 2., 0., 2., 6., 2.,
                                              0., 2., 6., 2., 0., 2., 6.],
                                             [6., 2., 0., 2., 6., 2., 0., 2., 6., 2., 0., 2., 6.,
                                              2., 0., 2., 6., 2., 0., 2.],
                                             [8., 6., 2., 0., 8., 6., 2., 0., 8., 6., 2., 0., 8.,
                                              6., 2., 0., 8., 6., 2., 0.],
                                             [0., 2., 6., 8., 0., 2., 6., 8., 0., 2., 6., 8., 0.,
                                              2., 6., 8., 0., 2., 6., 8.],
                                             [2., 0., 2., 6., 2., 0., 2., 6., 2., 0., 2., 6., 2.,
                                              0., 2., 6., 2., 0., 2., 6.],
                                             [6., 2., 0., 2., 6., 2., 0., 2., 6., 2., 0., 2., 6.,
                                              2., 0., 2., 6., 2., 0., 2.],
                                             [8., 6., 2., 0., 8., 6., 2., 0., 8., 6., 2., 0., 8.,
                                              6., 2., 0., 8., 6., 2., 0.],
                                             [0., 2., 6., 8., 0., 2., 6., 8., 0., 2., 6., 8., 0.,
                                              2., 6., 8., 0., 2., 6., 8.],
                                             [2., 0., 2., 6., 2., 0., 2., 6., 2., 0., 2., 6., 2.,
                                              0., 2., 6., 2., 0., 2., 6.],
                                             [6., 2., 0., 2., 6., 2., 0., 2., 6., 2., 0., 2., 6.,
                                              2., 0., 2., 6., 2., 0., 2.],
                                             [8., 6., 2., 0., 8., 6., 2., 0., 8., 6., 2., 0., 8.,
                                              6., 2., 0., 8., 6., 2., 0.],
                                             [0., 2., 6., 8., 0., 2., 6., 8., 0., 2., 6., 8., 0.,
                                              2., 6., 8., 0., 2., 6., 8.],
                                             [2., 0., 2., 6., 2., 0., 2., 6., 2., 0., 2., 6., 2.,
                                              0., 2., 6., 2., 0., 2., 6.],
                                             [6., 2., 0., 2., 6., 2., 0., 2., 6., 2., 0., 2., 6.,
                                              2., 0., 2., 6., 2., 0., 2.],
                                             [8., 6., 2., 0., 8., 6., 2., 0., 8., 6., 2., 0., 8.,
                                              6., 2., 0., 8., 6., 2., 0.]])

result_binify_even_bin_bw_maxlag = np.array([0., 0., 1.41421356, 0., 0.,
                                             0., 1.41421356, 0., 1.41421356, 0.])

result_group_to_bin = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
     3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
     3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
     3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
     3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [],
    [],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 2, 1, 2, 3, 2, 1, 2, 3,
     2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 2, 1, 0,
     1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 3, 2, 3, 2,
     3, 2, 3, 2, 3, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 2, 1,
     2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1,
     2, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2,
     1, 0, 1, 2, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2],
    [],
    [],
    [0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 3, 1, 3, 1, 3, 1, 3, 1,
     3, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 3, 1, 3, 1, 3, 1,
     3, 1, 3, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 3, 1, 3, 1,
     3, 1, 3, 1, 3, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 3, 1,
     3, 1, 3, 1, 3, 1, 3, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0,
     3, 1, 3, 1, 3, 1, 3, 1, 3, 1],
    [],
    [],
    [0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 0, 3, 0, 3, 0, 3, 0,
     3, 0, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 0, 3, 0, 3, 0,
     3, 0, 3, 0, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0]]

group_to_bin_bm = np.array([[0., 3., 6., 9., 0., 3., 6., 9., 0., 3., 6., 9., 0.,
                             3., 6., 9., 0., 3., 6., 9.],
                            [3., 0., 3., 6., 3., 0., 3., 6., 3., 0., 3., 6., 3.,
                             0., 3., 6., 3., 0., 3., 6.],
                            [6., 3., 0., 3., 6., 3., 0., 3., 6., 3., 0., 3., 6.,
                             3., 0., 3., 6., 3., 0., 3.],
                            [9., 6., 3., 0., 9., 6., 3., 0., 9., 6., 3., 0., 9.,
                             6., 3., 0., 9., 6., 3., 0.],
                            [0., 3., 6., 9., 0., 3., 6., 9., 0., 3., 6., 9., 0.,
                             3., 6., 9., 0., 3., 6., 9.],
                            [3., 0., 3., 6., 3., 0., 3., 6., 3., 0., 3., 6., 3.,
                             0., 3., 6., 3., 0., 3., 6.],
                            [6., 3., 0., 3., 6., 3., 0., 3., 6., 3., 0., 3., 6.,
                             3., 0., 3., 6., 3., 0., 3.],
                            [9., 6., 3., 0., 9., 6., 3., 0., 9., 6., 3., 0., 9.,
                             6., 3., 0., 9., 6., 3., 0.],
                            [0., 3., 6., 9., 0., 3., 6., 9., 0., 3., 6., 9., 0.,
                             3., 6., 9., 0., 3., 6., 9.],
                            [3., 0., 3., 6., 3., 0., 3., 6., 3., 0., 3., 6., 3.,
                             0., 3., 6., 3., 0., 3., 6.],
                            [6., 3., 0., 3., 6., 3., 0., 3., 6., 3., 0., 3., 6.,
                             3., 0., 3., 6., 3., 0., 3.],
                            [9., 6., 3., 0., 9., 6., 3., 0., 9., 6., 3., 0., 9.,
                             6., 3., 0., 9., 6., 3., 0.],
                            [0., 3., 6., 9., 0., 3., 6., 9., 0., 3., 6., 9., 0.,
                             3., 6., 9., 0., 3., 6., 9.],
                            [3., 0., 3., 6., 3., 0., 3., 6., 3., 0., 3., 6., 3.,
                             0., 3., 6., 3., 0., 3., 6.],
                            [6., 3., 0., 3., 6., 3., 0., 3., 6., 3., 0., 3., 6.,
                             3., 0., 3., 6., 3., 0., 3.],
                            [9., 6., 3., 0., 9., 6., 3., 0., 9., 6., 3., 0., 9.,
                             6., 3., 0., 9., 6., 3., 0.],
                            [0., 3., 6., 9., 0., 3., 6., 9., 0., 3., 6., 9., 0.,
                             3., 6., 9., 0., 3., 6., 9.],
                            [3., 0., 3., 6., 3., 0., 3., 6., 3., 0., 3., 6., 3.,
                             0., 3., 6., 3., 0., 3., 6.],
                            [6., 3., 0., 3., 6., 3., 0., 3., 6., 3., 0., 3., 6.,
                             3., 0., 3., 6., 3., 0., 3.],
                            [9., 6., 3., 0., 9., 6., 3., 0., 9., 6., 3., 0., 9.,
                             6., 3., 0., 9., 6., 3., 0.]])

result_group_to_bin_maxlag = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
     3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
     3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
     3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
     3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [], [],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 2, 1, 2, 3, 2, 1, 2, 3,
     2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 2, 1, 0,
     1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 3, 2, 3, 2,
     3, 2, 3, 2, 3, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 2, 1,
     2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1,
     2, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2,
     1, 0, 1, 2, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2], [], [], [],
    [0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 3, 1, 3, 1, 3, 1, 3, 1,
     3, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 3, 1, 3, 1, 3, 1,
     3, 1, 3, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 3, 1, 3, 1,
     3, 1, 3, 1, 3, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 3, 1,
     3, 1, 3, 1, 3, 1, 3, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0,
     3, 1, 3, 1, 3, 1, 3, 1, 3, 1], [],
    [0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 0, 3, 0, 3, 0, 3, 0,
     3, 0, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 0, 3, 0, 3, 0,
     3, 0, 3, 0, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0]]

result_group_to_bin_azimuth_and_tolerance = [
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0,
     0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
     1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
     2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
     3, 3, 3, 3, 3, 3, 3, 3], [], [], [], [], [], [], [], [], []]


@contextmanager
def get_stdout():
    outs, errs = StringIO(), StringIO()
    stdout, stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = outs, errs
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout = stdout
        sys.stderr = stderr


class TestBinifyEvenWidth(unittest.TestCase):
    def setUp(self):
        self.coordinates = np.array([(0, 0), (1, 1), (2, 2), (3, 3)] * 5)
        self.values = np.array([0, 1, 2, 3] * 5)

    def test_bm(self):
        """
        Testing binning matrix of binify_even_width Function
        """
        res = np.asarray(binify_even_width(self.coordinates)[0])

        assert_array_almost_equal(res[0],result_binify_even_width_bm)
        self.assertEqual(res.size, 400)
        self.assertEqual(res.shape, (20, 20))

    def test_bw(self):
        """
        Testing binning width array of binify_even_width Function
        """

        assert_array_almost_equal(np.asarray(binify_even_width(self.coordinates)[1]),
                                  result_binify_even_width_bw)

    def test_bm_n6(self):
        """
        Testing binning matrix of binify_even_width Function with 6 bins
        """
        res = np.asarray(binify_even_width(self.coordinates, N=6)[0])

        assert_array_almost_equal(res[0], result_binify_even_width_bm_n6)
        self.assertEqual(res.size, 400)
        self.assertEqual(res.shape, (20, 20))

    def test_bw_n6(self):
        """
        Testing binning width array of binify_even_width Function with 6 bins
        """

        assert_array_almost_equal(np.asarray(binify_even_width(self.coordinates, N=6)[1]),
                                  result_binify_even_width_bw_n6)

    def test_bm_with_dm(self):
        """
        Testing binning matrix of binify_even_width Function with a distance matrix
        """
        res = np.asarray(binify_even_width(self.coordinates, dm=binify_dm)[0])

        assert_array_almost_equal(res[0], result_binify_even_width_bm)
        self.assertEqual(res.size, 20)
        self.assertEqual(res.shape, (1, 20))

    def test_bm_maxlag(self):
        """
        Testing binning matrix of binify_even_width Function with a maxlag
        """
        res = np.asarray(binify_even_width(self.coordinates, maxlag=4)[0])

        assert_array_almost_equal(res[0], result_binify_even_width_bm_maxlag)
        self.assertEqual(res.size, 400)
        self.assertEqual(res.shape, (20, 20))

    def test_bw_maxlag(self):
        """
        Testing binning matrix of binify_even_width Function with a maxlag
        """

        assert_array_almost_equal(np.asarray(binify_even_width(self.coordinates, maxlag=4)[1]),
                                  result_binify_even_width_bw_maxlag)

    def test_misshaped_coordinates(self):
        """
        Pass misshaped coordinates
        """
        coords = list(self.coordinates)
        coords[2] = (2, 2, 2)
        with self.assertRaises(ValueError):
            binify_even_width(coords)

    def test_missing_arguments(self):
        """
        Provide too less arguments

        """
        with self.assertRaises(ValueError):
            binify_even_width(self.coordinates, N=None, w=None)

    def test_warning_too_much_arguments(self):
        """
        Raise a Warning, becasue too many arguments were passed
        """
        with get_stdout() as (out, err):
            binify_even_width(self.coordinates, N=5, w=1)
            self.assertEqual(
                out.getvalue().strip(),
                'Warning! w = 1 is ignored because N is already given'
            )


class TestBinifyEvenBin(unittest.TestCase):
    def setUp(self):
        self.coordinates = np.array([(0, 0), (1, 1), (2, 2), (3, 3)] * 5)
        self.values = np.array([0, 1, 2, 3] * 5)

    def test_bm(self):
        """
        Testing binning matrix of binify_even_bin Function
        """
        res = np.asarray(binify_even_bin(self.coordinates)[0])

        assert_array_almost_equal(res[0], result_binify_even_bin_bm)
        self.assertEqual(res.size, 400)
        self.assertEqual(res.shape, (20, 20))

    def test_bw(self):
        """
        Testing binning width array of binify_even_bin Function
        """

        assert_array_almost_equal(np.asarray(binify_even_bin(self.coordinates)[1]),
                                  result_binify_even_bin_bw)

    def test_bm_n6(self):
        """
        Testing binning matrix of binify_even_bin Function with 6 bins
        """
        res = np.asarray(binify_even_bin(self.coordinates, N=6)[0])

        assert_array_almost_equal(res[0], result_binify_even_bin_bm_n6)
        self.assertEqual(res.size, 400)
        self.assertEqual(res.shape, (20, 20))

    def test_bw_n6(self):
        """
        Testing binning width array of binify_even_bin Function with 6 bins
        """

        assert_array_almost_equal(np.asarray(binify_even_bin(self.coordinates, N=6)[1]),
                                  result_binify_even_bin_bw_n6)

    def test_bm_maxlag(self):
        """
        Testing binning matrix of binify_even_bin Function with a maxlag
        """

        assert_array_almost_equal(np.asarray(binify_even_bin(self.coordinates, maxlag=4)[0]),
                                  result_binify_even_bin_bm_maxlag)

    def test_bw_maxlag(self):
        """
        Testing binning width array of binify_even_bin Function with a maxlag
        """

        assert_array_almost_equal(np.asarray(binify_even_bin(self.coordinates, maxlag=4)[1]),
                                  result_binify_even_bin_bw_maxlag)

    def test_misshaped_coordinates(self):
        """
        Pass misshaped coordinates
        """
        coords = list(self.coordinates)
        coords[2] = (2, 2, 2)
        with self.assertRaises(ValueError):
            binify_even_bin(coords)


class TestGrouToBin(unittest.TestCase):
    def setUp(self):
        self.coordinates = np.array([(0, 0), (1, 1), (2, 2), (3, 3)] * 5)
        self.values = np.array([0, 1, 2, 3] * 5)


    def test_default(self):
        """
        Testing group_to_bin function
        """

        self.assertEqual(group_to_bin(self.values, X=self.coordinates), result_group_to_bin)

    def test_with_bm(self):
        """
        Testing group_to_bin function
        """

        self.assertEqual(group_to_bin(self.values, bm=group_to_bin_bm), result_group_to_bin)

    def test_with_bm(self):
        """
        Testing group_to_bin function
        """

        self.assertEqual(group_to_bin(self.values, bm=group_to_bin_bm), result_group_to_bin)

    def test_with_maxlag(self):
        """
        Testing group_to_bin function with maxlag
        """

        self.assertEqual(group_to_bin(self.values, X=self.coordinates, maxlag=4), result_group_to_bin_maxlag)

    def test_with_azimuth_and_tolerance(self):
        """
        Testing group_to_bin function with azimuth and tolerance
        """

        self.assertEqual(group_to_bin(self.values, X=self.coordinates, azimuth_deg=90, tolerance=22.5),
                         result_group_to_bin_azimuth_and_tolerance)


class TestInBounds(unittest.TestCase):
    def setUp(self):
        self.coordinates = np.array([(0, 0), (1, 1), (2, 2), (3, 3)] * 5)
        self.values = np.array([0, 1, 2, 3] * 5)

    def test_in_bounds(self):
        """        Testing in_bounds function with result = True
        """

        self.assertTrue(_in_bounds(50, 40, 10))
        self.assertFalse(_in_bounds(50, 10, 40))


if __name__ == '__main__':
    unittest.main()
