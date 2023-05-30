'''NWTW: Needleman-Wunsch Time Warping

Code is from https://github.com/HMC-MIR/FlexDTW/blob/main/NWTW.ipynb and http://phenicx.upf.edu/system/files/publications/mgrachten-et-al-ISMIR2013.pdf
'''
import numpy as np
from numba import jit

@jit(nopython=True)
def nwtw_backtrace(D, B, steps):
    '''Backtraces through the cumulative cost matrix D starting from a specified location.
    
    Args:
        D: cumulative cost matrix
        B: backtrace matrix
        steps: a numpy matrix specifying the allowable transitions.  It should be of dimension (L, 2), where each row specifies (row step, col step)
        rstart: the row index to start backtracking from
        cstart: the column index to start backtracking from
    
    Returns:
        A numpy array of (row, col) coordinates for the optimal path.
    '''

    path = []

    pos = (D.shape[0] - 1, D.shape[1] - 1)
    path.append(pos)
    while (pos != (0,0)):
        (row, col) = pos
        stepidx = B[row, col]
        (rstep, cstep) = steps[stepidx]
        pos = (row-rstep, col-cstep)
        path.append(pos)
    
    return path

@jit(nopython=True)
def nwtw(C, gamma=0.346):
    ''' Needleman-Wunsch Time Warping (NWTW)

    Args:
        C: cost matrix
        gamma: the cost of a gap

    Returns:
        D: cumulative cost matrix
        B: backtrace matrix
        path: a numpy array of (row, col) coordinates for the optimal path.
        optimal_cost: the optimal cost

    '''
    D = np.zeros(C.shape)
    B = np.zeros(C.shape, dtype=np.int8)

    # Allowed steps as specified by the paper
    steps = np.array([0, 1, 1, 0, 1, 1, 1, 2, 2, 1]).reshape((5,2))

    # initialize -- makes inner for loop more efficient
    D[:,0] = gamma * np.arange(C.shape[0])
    B[:,0] = 1
    D[0,:] = gamma * np.arange(C.shape[1])
    B[0,:] = 0
    D[0,0] = 0 # in case gamma = np.inf

    for i in range(1, C.shape[0]):
        for j in range(1, C.shape[1]):
            # Min bracket
            mincost = np.inf
            minidx = -1
            for stepidx, step in enumerate(steps):
                (rstep, cstep) = step
                prevrow = i - rstep
                prevcol = j - cstep
                if prevrow >= 0 and prevcol >= 0:
                    pathcost = D[prevrow, prevcol]
                    # nw(i - 1, j - 2)
                    if (stepidx == 3):
                        pathcost += C[i, j - 1] + C[i, j]
                    # nw(i - 2, j - 1)
                    elif (stepidx == 4):
                        pathcost += C[i - 1, j] + C[i, j]
                    # nw(i - 1, j - 1)
                    elif (stepidx == 2):
                        pathcost += C[i, j]
                    else:
                        pathcost += gamma
                    if pathcost < mincost:
                        mincost = pathcost
                        minidx = stepidx
            # TT: is this if clause necessary with the for loops starting from 1?  I think it can be removed
            if minidx == -1:
                raise Exception("NWTW edge case error with -1 indexes")
            D[i, j] = mincost
            B[i, j] = minidx

    optcost = D[-1,-1]
    path = nwtw_backtrace(D, B, steps)
    path.reverse()

    path = np.array(path).T

    return D, B, path, optcost