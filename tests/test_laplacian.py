import warnings

import numpy.testing as npt
import pytest
import warnings

from cued_sf2_lab.laplacian_pyramid import *

try:
    from cued_sf2_lab.answers.laplacian_pyramid import *
except ImportError:
    warnings.warn('Answer directory failed to import')


def test_rowdec_odd():
    X = np.array(
        [[0, 0, 0, 0],
         [2, 1, 4, 1]])
    Y1 = rowdec(X, [1, 2, 1])
    npt.assert_equal(Y1,
        [[0, 0],
         [6, 10]])
    Y2 = rowdec2(X, [1, 2, 1])
    npt.assert_equal(Y2,
        [[0, 0],
         [8, 10]])

def test_rowdec_even():
    X = np.array(
        [[0, 0, 0, 0],
         [2, 1, 4, 1]])
    Y1 = rowdec(X, [1, 1])
    npt.assert_equal(Y1,
        [[0, 0],
         [3, 5]])

# TODO: fix this!
@pytest.mark.xfail(exception=IndexError)
def test_rowdec2_even():
    X = np.array(
        [[0, 0, 0, 0],
         [2, 1, 4, 1]])
    Y2 = rowdec2(X, [1, 1])
    npt.assert_equal(Y2,
        [[0, 0],
         [8, 10]])

def test_image_dec():
    X = np.array(
        [[0, 0, 0, 0],
         [2, 1, 4, 1]])
    Y = image_dec(X, [1, 2, 1])
    npt.assert_equal(Y, [[12, 20]])

# TODO: test `test_plot_laplacian_pyramid`

def test_beside():
    a = np.full((5, 5), 1)
    b = np.full((2, 2), 2)
    c = np.full((1, 1), 2)
    abc = beside(a, beside(b, c))
    npt.assert_equal(abc,
        [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 0, 2, 2, 0, 2],
         [1, 1, 1, 1, 1, 0, 2, 2, 0, 0],
         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])

def test_rowint_odd():
    X = np.array(
        [[0, 0, 0, 0],
         [2, 1, 4, 1]])
    Y1 = rowdec(X, [1, 2, 1])
    Z = rowint(Y1, [1, 2, 1])
    npt.assert_equal(Z,
        [[ 0,  0,  0,  0],
         [12, 16, 20, 20]])

def test_rowint_even():
    X = np.array(
        [[0, 0, 0, 0],
         [2, 1, 4, 1]])
    Y1 = rowdec(X, [1, 1])
    Z = rowint(Y1, [1, 1])
    npt.assert_equal(Z,
        [[0, 0, 0, 0],
         [6, 3, 5, 5]])

def test_image_int():
    X = np.array(
        [[0, 0, 0, 0],
         [2, 1, 4, 1]])
    Y = image_dec(X, [1, 2, 1])
    Z = image_int(Y, [1, 2, 1])
    npt.assert_equal(Z,
        [[192, 256, 320, 320],
         [192, 256, 320, 320]])

# TODO: write the rest of these

def test_py4enc():
    pass

def test_py4dec():
    pass

def test_quant1():
    pass

def test_quant2():
    pass

def test_quantise():
    pass

def test_bpp():
    pass
