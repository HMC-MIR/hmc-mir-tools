'''FlexDTW: Dynamic Time Warping with Boundaries Constraint Relaxation

Code is from: https://github.com/HMC-MIR/FlexDTW/blob/main/FlexDTW.ipynb and
'''

import numpy as np
from numba import jit

@jit(nopython=True)
def find_best_endpoint(D, P, buffer):
    '''Determines the best location to begin backtracking from by comparing the average path cost
    per manhattan block.
    
    Args:
        D: the cumulative cost matrix
        P: the matrix specifying the starting location of the alignment path
        buffer: specifies the length of a buffer region (in frames) to avoid short degenerate alignment paths
            near the corners of the pairwise cost matrix.  This can be thought of as the minimum length that
            needs to match in order to be considered a valid alignment path.
    
    Returns:
        best_cost: the best average path cost per manhattan block
        best_r: the row index of the best endpoint
        best_c: the column index of the best endpoint
        debug: debugging information for examining the average cost per manhattan block for each of the candidate ending positions
    '''
    
    # consider last row and column as candidates
    candidates = [(D.shape[0]-1,i) for i in range(buffer, D.shape[1])] + [(i, D.shape[1]-1) for i in range(buffer, D.shape[0]-1)][::-1]
    
    best_cost = np.inf
    best_r, best_c = -1, -1
    debug = []
    
    for i, (r,c) in enumerate(candidates):
                
        # get alignment start location
        if P[r,c] >= 0:
            rstart, cstart = 0, P[r,c]
        else:
            rstart, cstart = -P[r,c], 0
            
        # calculate average cost per manhattan block
        mdist = (r - rstart) + (c - cstart) # manhattan distance
        avg_cost_per_mb = D[r,c] / mdist
        
        # keep best
        if avg_cost_per_mb < best_cost:
            best_cost = avg_cost_per_mb
            best_r, best_c = r, c
            
        # debugging info
        if r == D.shape[0]-1:
            debug.append((c-D.shape[1]+1, avg_cost_per_mb, r, c))
        else:
            debug.append((D.shape[0]-1-r, avg_cost_per_mb, r, c))
    
    return best_cost, best_r, best_c, debug

@jit(nopython=True)
def flex_dtw_backtrace(D, B, steps, rstart, cstart):
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
    pos = (rstart, cstart)
    path = []
    path.append(pos)
    while(pos[0] != 0 and pos[1] != 0):
        (row, col) = pos
        stepidx = B[row, col]
        (rstep, cstep) = steps[stepidx]
        pos = (row-rstep, col-cstep)
        path.append(pos)
    
    return path

@jit(nopython=True)
def flex_dtw(C, steps, weights, buffer = 1):
    '''Dynamic Time Warping

    Args:
        C: a numpy matrix of pairwise costs
        steps: a numpy matrix specifying the allowable transitions.  It should be of dimension (L, 2), where each row specifies (row step, col step)
        weights: a numpy array of length L specifying the weight for each step
        buffer: specifies the length of a buffer region (in frames) to avoid short degenerate alignment paths

    Returns:
        D: cumulative cost matrix
        B: backtrace matrix
        path: numpy array of (row, col) coordinates for the optimal path.
        optimal_cost: the optimal cost
        debug: debugging information for examining the average cost per manhattan block for each of the candidate ending positions (see find_best_endpoint)
    '''
    
    # initialize
    D = np.zeros(C.shape)
    B = np.zeros(C.shape, dtype=np.int8)
    P = np.zeros(C.shape, dtype=np.int32)
    D[0,:] = C[0,:]
    D[:,0] = C[:,0]
    
    # DP
    for row in range(1,C.shape[0]):
        for col in range(1, C.shape[1]):
            mincost = np.inf
            minidx = -1
            bestrprev = -1
            bestcprev = -1
            for stepidx, step in enumerate(steps):
                (rstep, cstep) = step
                prevrow = row - rstep
                prevcol = col - cstep
                if prevrow >= 0 and prevcol >= 0:
                    
                    # calculate avg cost per manhattan block
                    pathcost = D[prevrow, prevcol] + C[row, col] * weights[stepidx]
                    if P[prevrow, prevcol] >= 0:
                        mdist = row + (col - P[prevrow, prevcol])
                    else:
                        mdist = (row + P[prevrow, prevcol]) + col
                    cost_per_mb = pathcost / mdist
                    
                    # select best transition based on avg cost per manhattan block
                    if cost_per_mb < mincost:
                        mincost = cost_per_mb
                        minidx = stepidx
                        bestrprev = prevrow
                        bestcprev = prevcol
                        
            D[row, col] = D[bestrprev, bestcprev] + C[row, col] * weights[minidx]
            B[row, col] = minidx
            if bestrprev == 0:
                P[row, col] = bestcprev
            elif bestcprev == 0:
                P[row, col] = -1*bestrprev
            else:
                P[row, col] = P[bestrprev, bestcprev]
            
    #  backtrack
    best_cost, best_r, best_c, debug = find_best_endpoint(D, P, buffer)
    path = flex_dtw_backtrace(D, B, steps, best_r, best_c)
    path.reverse()
    path = np.array(path)
    
    return D, B, path.T, best_cost, debug