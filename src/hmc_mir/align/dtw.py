'''Standard DTW

Code is from https://github.com/HMC-MIR/FlexDTW/blob/main/DTW.ipynb
'''
import numpy as np
from numba import jit

@jit(nopython=True)
def dtw_backtrace(D, B, steps, subseq=False):
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

    rstart = B.shape[0] - 1
    if subseq:
        cstart = np.argmin(D[-1])
    else:
        cstart = B.shape[1] - 1
    pos = (rstart, cstart)
    path = []
    path.append(pos)
    while (pos[0] != 0 and pos[1] != 0) or (pos[0] and subseq):
        (row, col) = pos
        stepidx = B[row, col]
        (rstep, cstep) = steps[stepidx]
        pos = (row-rstep, col-cstep)
        path.append(pos)
    
    return path

@jit(nopython=True)
def dtw(C, steps, weights, subseq=False):
    '''Standard DTW
    
    Args:
        C: a numpy matrix of pairwise costs
        steps: a numpy matrix specifying the allowable transitions.  It should be of dimension (L, 2), where each row specifies (row step, col step)
        weights: a numpy array of length L specifying the weight for each step
        subseq: if True, then the optimal path is constrained to be a subsequence of the input sequences

    Returns:
        D: cumulative cost matrix
        B: backtrace matrix
        path: a numpy array of (row, col) coordinates for the optimal path.
    '''
    D = np.ones(C.shape) * np.inf
    B = np.zeros(C.shape, dtype=np.int8)

    if subseq:
        D[0, :] = C[0,:]
    else:
        D[0, 0] = C[0,0]

    for row in range(C.shape[0]):
        for col in range(C.shape[1]):
            bestCost = D[row, col]
            bestCostIndex = -1
            for stepIndex in range(steps.shape[0]):
                if row - steps[stepIndex][0] >= 0 and col - steps[stepIndex][1] >= 0:
                    costForStep = C[row, col] * weights[stepIndex] + D[row - steps[stepIndex][0], col - steps[stepIndex][1]]
                    if costForStep < bestCost:
                        bestCost = costForStep
                        bestCostIndex = stepIndex
            D[row, col] = bestCost
            B[row, col] = bestCostIndex
    
    path = dtw_backtrace(D, B, steps, subseq)
    path.reverse()
    path = np.array(path).T

    return D, B, path