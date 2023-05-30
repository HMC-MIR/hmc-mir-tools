'''Hidden State Time Warping

Code is from https://github.com/HMC-MIR/HSTW and https://www.mdpi.com/1579848
'''
import numpy as np
import numba
from sklearn.metrics.pairwise import euclidean_distances

def nwtwdp_backtrace3D(B, D):
    '''Backtrace the optimal path from the cumulative cost matrix and the backtrace matrix.
    
    Args:
        B: the backtrace matrix
        D: the cumulative cost matrix

    Returns:
        A numpy array of (x, y, z) coordinates for the optimal path.
    '''
    p = 0
    r = D.shape[1] - 1
    c = np.argmin(D[0, -1])
    path_3D = []
    while r > 0:
        path_3D.append([p,r,c])
        if B[p, r, c] == 0 and p == 0:
            p = 1
            r -= 1
            c -= 1
        elif B[p, r, c] == 0 and p == 1:
            p = 0
        elif B[p, r, c] == 1:
            r -= 1
            c -= 1
        elif B[p, r, c] == 2:
            c -= 1
        elif B[p, r, c] == 3:
            r -= 1
    return np.asarray(path_3D)

@numba.jit(nopython=True)
def nwtwdp(C, alpha, beta=20, gamma = 1):
    # 0: visible, 1: hidden
    # B: 1 Diag, 2 Right, 3 Up, 0 switch plane
    # initialization
    D = np.zeros((2, C.shape[0], C.shape[1]))
    B = np.zeros((2, C.shape[0], C.shape[1]))
    
    # bottom rows
    D[0, 0, :] = C[0, :]
    D[1, 0, :] = np.inf
    
    # first cols
    for i in range(1, C.shape[0]):
        D[0, i, 0] = D[0, i-1, 0] + alpha
        D[1, i, 0] = D[0, i, 0]
        B[0, i, 0] = 3
        B[1, i, 0] = 0
        
    # rest of the matrix
    for i in range(1, C.shape[0]):
        for j in range(1, C.shape[1]):
        
            # hidden
            # diag visible -> hidden, right in hidden, up in hidden
            costs = np.array([D[0, i-1, j-1] + gamma + alpha, np.inf, D[1, i, j-1] + gamma, D[1, i-1, j] + alpha])
            D[1, i, j] = np.min(costs)
            B[1, i, j] = np.argmin(costs)
                
            # visible
            # hidden -> visible, diag
            costs = np.array([D[1, i, j] + beta, D[0, i-1, j-1] + C[i, j]])
            D[0, i, j] = np.min(costs)
            B[0, i, j] = np.argmin(costs)
            
    return B, D

def hstw(mfcc_ref, mfcc_query, Ca = 2.4, Cb = 33, gamma = 3):
    '''Aligns a query file with its corresponding reference file and returns the 3-D path through the HSTW tensor

    Args:
        mfcc_ref (np.ndarray): MFCCs of the reference file
        mfcc_query (np.ndarray): MFCCs of the query file
        Ca (float, optional): Cost of a hidden state. Defaults to 2.4.
        Cb (float, optional): Cost of a visible state. Defaults to 33.
        gamma (int, optional): Cost of a diagonal transition. Defaults to 3.
    
    Returns:
        D: cumulative cost matrix
        B: backtrace matrix
        path: a numpy array of (x, y, z) coordinates for the optimal path.
    '''
    C = euclidean_distances(mfcc_query, mfcc_ref)
    alpha = np.median(np.min(C, axis=1)) * Ca
    B, D = nwtwdp(C, alpha, beta=(alpha+gamma)*Cb)
    path_3D = nwtwdp_backtrace3D(B, D)
    return D, B, path_3D, C