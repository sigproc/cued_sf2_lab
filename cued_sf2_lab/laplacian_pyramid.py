import scipy.io
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from matplotlib import colors

from .familiarisation import load_mat_img
from .familiarisation import prep_cmap_array_plt
from .familiarisation import plot_image


def rowdec(X, h):
    """
    Filter rows of image X with h and then decimate by a factor of 2.

    Parameters:
    X (numpy.ndarray): Image matrix (Usually 256x256)
    h (numpy.ndarray): Filter coefficients
    Returns:
    Y (numpy.ndarray): Image with filtered and decimated rows

    If len(H) is odd, each output sample is aligned with the first of
    each pair of input samples.
    If len(H) is even, each output sample is aligned with the mid point
    of each pair of input samples.
    """
    [r, c] = X.shape
    m = len(h)
    m2 = int(np.fix((m)/2))
    if (m % 2) > 0:
        # Odd h: symmetrically extend indices without repeating end samples.
        xe = np.array(list(range(m2, 0, -1)) + list(range(0, c)) +
                      list(range(c-2, c-m2-2, -1)), dtype=int)
        # print(xe)
    else:
        # Even h: symmetrically extend with repeat of end samples.
        xe = np.array(list(range(m2-2, -1, -1)) + list(range(0, c)) +
                      list(range(c-1, c-m2, -1)), dtype=int)
        # print(xe)

    t = np.array(range(0, c, 2), dtype=int)
    # print(t)
    Y = np.zeros((r, len(t)))
    # print(Y)
    # Loop for each term in h.
    # print(X)
    for i in range(0, m):
        # print(i) --> 0 1 2
        #print(f'hello %d', i)
        #print(h[i])
        #print(X[:, xe[t+i]])
        #print(h[i] * X[:, xe[t+i]])
        Y = Y + h[i] * X[:, xe[t+i]]
    # print(X)
    return Y


# TODO: FIX this - breaks for even filters (like MATLAB function)
def rowdec2(X, h):
    """
    Filter rows of image X with h and then decimate by a factor of 2.

    Parameters:
    X (numpy.ndarray): Image matrix (Usually 256x256)
    h (numpy.ndarray): Filter coefficients
    Returns:
    Y (numpy.ndarray): Image with filtered and decimated rows

    If len(H) is odd, each output sample is aligned with the second of
    each pair of input samples.
    If len(H) is even, each output sample is aligned with the mid point
    of each pair of input samples.


    """
    [r, c] = X.shape
    m = len(h)
    m2 = int(np.fix((m)/2))
    if (m % 2) > 0:
        # Odd h: symmetrically extend indices without repeating end samples.
        xe = np.array(list(range(m2, 0, -1)) + list(range(0, c)) +
                      list(range(c-2, c-m2-2, -1)), dtype=int)
        # print(xe)
    else:
        # Even h: symmetrically extend with repeat of end samples.
        xe = np.array(list(range(m2-2, -1, -1)) + list(range(0, c)) +
                      list(range(c-1, c-m2, -1)), dtype=int)
        # print(xe)

    t = np.array(range(1, c, 2), dtype=int)
    # print(t)
    Y = np.zeros((r, len(t)))
    # print(Y)
    # Loop for each term in h.
    for i in range(0, m):
        # print(i) --> 0 1 2
        #print(f'hello %d', i)
        #print(h[i])
        #print(X[:, xe[t+i]])
        #print(h[i] * X[:, xe[t+i]])
        Y = Y + h[i] * X[:, xe[t+i]]
    return Y


# Something like `axs = plt.subplots(5, sharex=True, sharey=True)`
# TODO: Use beside function several times
def plot_laplacian_pyramid(X, decimated_list):
    """
    Plot laplacian pyramid images side by side.

    Parameters:
    X (numpy.ndarray): Original image matrix (Usually 256x256)
    decimated_list (list): List of X1, X2 etc
    """
    plot_list = [X]
    for X_dec in decimated_list:
        X_dec_padded = np.zeros_like(X)
        X_dec_padded[:X_dec.shape[0], :X_dec.shape[1]] = X_dec
        plot_list.append(X_dec_padded)

    plot_image(np.hstack(tuple(plot_list)))


# TODO: Fixup
def beside(X1, X2):
    """
    Arrange two images beside eachother.

    Parameters:
    X1, X2 (numpy.ndarray): Original image matrices (Usually 256x256)

    Returns:
    Y (numpy.ndarray): Padded with zeros as necessary and the images are
    separated by a blank column
    """
    [m1, n1] = X1.shape
    [m2, n2] = X2.shape
    # print(m1,n1,m2,n2)
    m = max(m1, m2)
    Y = np.zeros((m, n1+n2+1))
    # print(Y.shape)
    # print(((m-m1)/2)+1)
    # print(type(n1))

    # index slicing must use integers
    Y[int(((m-m1)/2)):int(((m-m1)/2)+m1), :n1] = X1
    Y[int(((m-m2)/2)):int(((m-m2)/2)+m2), n1+1:n1+1+n2] = X2

    return Y


def rowint(X, h):
    """
    Interpolates the rows of image X by 2 using h.

    Parameters:
    X (numpy.ndarray): Image matrix (Usually 256x256)
    h (numpy.ndarray): Filter coefficients
    Returns:
    Y (numpy.ndarray): Image with interpolated rows

    If len(h) is odd, each input sample is aligned with the first of
    each pair of output samples.
    If len(h) is even, each input sample is aligned with the mid point
    of each pair of output samples.
    """
    [r, c] = X.shape
    m = len(h)
    m2 = int(np.fix((m)/2))
    c2 = 2 * c
    if (m % 2) > 0:
        xe = np.array(list(range(m2, 0, -1)) + list(range(0, c2)) +
                      list(range(c2-2, c2-m2-2, -1)), dtype=int)
    else:
        xe = np.array(list(range(m2-1, -1, -1)) + list(range(0, c2)) +
                      list(range(c2-1, c2-m2-1, -1)), dtype=int)

    t = np.array(range(0, c2), dtype=int)

    # Generate X2 as X interleaved with columns of zeros.
    X2 = np.zeros((r, c2))
    X2[:, range(0, c2, 2)] = X.copy()

    Y = np.zeros((r, c2))
    # Loop for each term in h.
    for i in range(0, m):
        Y = Y + h[i] * X2[:, xe[t+i]]
    return Y


def quant1(x, step, rise1=None):
    """
    Quantise the matrix x using steps of width step.

    The result is the quantised integers Q. If rise1 is defined,
    the first step rises at rise1, otherwise it rises at step/2 to
    give a uniform quantiser with a step centred on zero.
    In any case the quantiser is symmetrical about zero.
    """
    if step <= 0:
        q = x.copy()
        return q
    if rise1 is None:
        rise = step/2.0
    else:
        rise = rise1
    # Quantise abs(x) to integer values, and incorporate sign(x)..
    temp = np.ceil((np.abs(x) - rise)/step)
    q = temp*(temp > 0)*np.sign(x)
    return q


def quant2(q, step, rise1=None):
    """
    Reconstruct matrix Y from quantised values q using steps of width step.

    The result is the reconstructed values. If rise1 is defined, the first
    step rises at rise1, otherwise it rises at step/2 to give a uniform
    quantiser with a step centred on zero.
    In any case the quantiser is symmetrical about zero.
    """
    if step <= 0:
        y = q.copy()
        return y
    if rise1 is None:
        rise = step/2.0
        return q * step
    else:
        rise = rise1
        # Reconstruct quantised values and incorporate sign(q).
        y = q * step + np.sign(q) * (rise - step/2.0)
        return y


def quantise(x, step, rise1=None):
    """
    Quantise matrix x in one go with step width of step using quant1 and quant2

    If rise1 is defined, the first step rises at rise1, otherwise it rises at
    step/2 to give a uniform quantiser with a step centred on zero.
    In any case the quantiser is symmetrical about zero.
    """
    if step <= 0:
        y = x.copy()
        return y
    if rise1 is None:
        rise = step/2.0
    else:
        rise = rise1
    # Perform both quantisation steps
    y = quant2(quant1(x, step, rise), step, rise)
    return y


def bpp(x):
    """
    Calculate the entropy in bits per element (or pixel) for matrix x

    The entropy represents the number of bits per element to encode x
    assuming an ideal first-order entropy code.
    """
    minx = np.min(np.min(x))
    maxx = np.max(np.max(x))
    # Calculate histogram of x in bins defined by bins.
    bins = list(range(int(np.floor(minx))-1, int(np.ceil(maxx)+2)))
    if (np.ceil(maxx) - np.floor(minx)) < 2:
        # in this case there is no information, as all the values are identical
        b = 0
        return b
    else:
        [h, s] = np.histogram(x[:], bins)
    # bar(s,h)
    # figure(gcf)

    # Convert bin counts to probabilities, and remove zeros.
    p = h / np.sum(h)
    p = p[(p > 0).ravel().nonzero()]

    # Calculate the entropy of the histogram using base 2 logs.
    b = -np.sum(p * np.log(p)) / np.log(2)
    return b


if __name__ == "__main__":
    # testing of rowdec for dwt
    X = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
    h1 = np.array((-1/4,-2/4,-1/4, -1/4)) 
    h2=np.array((-1/4, -2/4, -1/4)) 
    #print(rowdec2(X, h1))
    print(rowdec(X, h2))
    '''
    h = 0.25*np.array([1, 2, 1])

    img = 'lighthouse.mat'
    img_info = 'X'
    cmap_info = {'map', 'map2'}
    X_pre_zero_mean, cmaps_dict = load_mat_img(img, img_info, cmap_info)
    X = X_pre_zero_mean - 128.0

    Y0, Y1, Y2, Y3, X4 = py4enc(X, h)
    # plot_laplacian_pyramid(Y0, [Y1, Y2, Y3, X4])

    plot_image(beside(Y0, beside(Y1, beside(Y2, beside(Y3, X4)))))

    Z3, Z2, Z1, Z0 = py4dec(Y0, Y1, Y2, Y3, X4, h)
    print('Max difference between X and Z0: ', np.max(np.abs(X-Z0)))

    # plot_image(beside(Z0,beside(Z1,beside(Z2,Z3))))
    # plot_laplacian_pyramid(Z0, [Z1,Z2,Z3])
    '''
    """
    X1 = image_dec(X, h)
    step = 17
    X_entropy_per_pixel = bpp(quantise(X,step))
    X1_entropy_per_pixel = bpp(quantise(X1,step))
    Y0_entropy_per_pixel = bpp(quantise(Y0,step))
    print('X_entropy_per_pixel ', X_entropy_per_pixel)
    print('X1_entropy_per_pixel ', X1_entropy_per_pixel)
    print('Y0_entropy_per_pixel ', Y0_entropy_per_pixel)

    X_total_entropy = X_entropy_per_pixel*np.prod(X.shape)
    X1_total_entropy = X1_entropy_per_pixel*np.prod(X1.shape)
    Y0_total_entropy = Y0_entropy_per_pixel*np.prod(Y0.shape)
    print('\n')
    print('X_total_entropy ', X_total_entropy)
    print('X1_total_entropy ', X1_total_entropy)
    print('Y0_total_entropy ', Y0_total_entropy)
    print(X_total_entropy/(X1_total_entropy+Y0_total_entropy))
    """
    # X1 = image_dec(X, h)
    # X2 = image_dec(X1,h)
    # plot_laplacian_pyramid(X, [X1,X2])
    # X1_padded = np.zeros_like(X)
    # X1_padded[:X1.shape[0], :X1.shape[1]] = X1
    # print(X1_padded.shape)
    # print(X.shape)
    # X1.resize(X.shape)
    # plot_image(X1)
    # plot_image(X)
    # plot_image(np.hstack((X,X1_padded)))
