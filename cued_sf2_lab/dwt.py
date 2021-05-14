import numpy as np

from .laplacian_pyramid import rowdec
from .laplacian_pyramid import rowdec2


# TODO complete this!
def dwt(X, h1=np.array([-1, 2, 6, 2, -1])/8, h2=np.array([-1, 2, -1])/4):
    """
    Return a 1-level 2-D discrete wavelet transform of X.

    Default h1 and h2 are the LeGall filter pair.

    Parameters:
    X (numpy.ndarray): Image matrix (Usually 256x256)
    h1 and h2 (numpy.ndarray): Filter coefficients
    Returns:
    Y (np.ndarray): 1-level 2D DWT of X
    """
    m, n = X.shape
    Y = np.zeros((m, n))

    n2 = int(n/2)
    # t = [a for a in range(int(n2))]
    Y[:, 0:(n2)] = rowdec(X, h1)
    #print(Y)
    # print(rowdec2(X, h2))
    Y[:, (0+n2):(n2+n2)] = rowdec2(X, h2)
    #print(Y)
    # NEED to do a copy of Y X=Y.T changes X everytime Y changes
    Y_new = np.copy(Y)
    X = Y_new.T
    #print(X)
    m2 = int(m/2)
    Y[0:(m2), :] = (rowdec(X, h1)).T
    # print(Y)
    #print(X)
    #print((rowdec2(X, h2)).T)
    Y[(0+m2):(m2+m2), : ] = (rowdec2(X, h2)).T
    #print(X)

    return Y

if __name__ == '__main__':
    X = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
    print(dwt(X))
