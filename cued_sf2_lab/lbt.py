import numpy as np

from .dct import dct_ii, dct_iv


def pot_ii(N, s=(1+(5**0.5))/2, overlap='default'):
    """
    Generates the 1-D POT transform matrices of size N

    Equivalent to the pre-filtering stage of a Type-II fast
    Lapped Orthogonal Transform (LOT-II).

    POT_II Photo Overlap Transform matrix

    Y = Pf * X pre-filters N-vector X into Y.
    X = Pr' * Y post-filters N-vector Y into X.

    Parameters:
    N (int): Size of DCT used
    s (float): Scaling factor determining the orthogonality of the transform.
    s=1 generates a LOT (Pr = Pf), otherwise 1<s<2 generates
    an LBT. The default is the Golden Ratio, (1+5^0.5)/2.
    overlap (int): Determines amount of overlap. 0<overlap<=N/2.
    Default is N/2, which implies complete overlap with the corresponding DCT

    Returns:
    Pf (np.ndarray): Prefiltering matrix
    Pr (np.ndarray): Postfiltering matrix

    """
    # ensure N is divisible 2 and is an integer
    if N % 2 != 0 or type(N) != int:
        raise ValueError('N must be an integer divisible by 2.')
    if overlap == 'default':
        # produces an integer which survives next test
        overlap = N//2
    # overlap must be an integer to create a matrix with it as dimensions
    if type(overlap) != int:
        raise ValueError('overlap must be an int.')
    # TODO: Ask Joan whether the lower limit is 0 or 1. She said 0 for now.
    if overlap > N/2 or overlap < 0:
        raise ValueError('overlap must satisfy 0<overlap<=N/2')
    # generate identity matrix
    Id = np.identity(N//2)
    # print('I', Id)
    # flip identity in the left/right direction
    J = np.fliplr(Id)

    Z = np.zeros((N//2, N//2))
    C_ii = dct_ii(overlap)
    # print('C_ii', C_ii)
    C_iv = dct_iv(overlap)
    # print('C_iv', C_iv)

    # generate forward and reverse scaling matrices
    # use a list to be able to concatenate
    diag_ones = [1 for i in range(overlap-1)]
    Sf = np.diag([s]+diag_ones)
    Sr = np.diag([1/s]+diag_ones)
    # print('Sf', Sf)
    # print('Sr', Sr)

    # generate forward and reverse filtering matrices
    if overlap < N/2:
        VI = np.identity((N//2)-overlap)
        VJ = np.fliplr(VI)
        VZ = np.zeros((overlap, (N//2)-overlap))
        # create intermediate matrices to stack together
        # for Vf
        vf_1 = np.hstack((VJ @ C_ii.T @ Sf @ C_iv @ VJ, VZ))
        vf_2 = np.hstack((VZ.T, VI))
        Vf = np.vstack((vf_1, vf_2))
        # for Vr
        vr_1 = np.hstack((VJ @ C_ii.T @ Sr @ C_iv @ VJ, VZ))
        vr_2 = np.hstack((VZ.T, VI))
        Vr = np.vstack((vr_1, vr_2))

    else:
        Vf = J @ C_ii.T @ Sf @ C_iv @ J
        Vr = J @ C_ii.T @ Sr @ C_iv @ J
    # create component matrices to build Pf and Pr
    mtrx_1 = np.vstack((np.hstack((Id, J)), np.hstack((J, -Id))))
    pf_1 = np.vstack((np.hstack((Id, Z)), np.hstack((Z, Vf))))
    pr_1 = np.vstack((np.hstack((Id, Z)), np.hstack((Z, Vr))))

    Pf = 0.5*(mtrx_1 @ pf_1 @ mtrx_1)
    Pr = 0.5*(mtrx_1 @ pr_1 @ mtrx_1)
    return Pf, Pr


if __name__ == '__main__':
    print(pot_ii(8))
