'''Iterative Subtractive Alignment

Code is from https://github.com/HMC-MIR/PianoTrioAlignment and https://archives.ismir.net/ismir2021/paper/000101.pdf
'''
from numba import njit, prange
import numpy as np
from skimage.filters import threshold_li, threshold_niblack, threshold_triangle, threshold_isodata, threshold_mean, threshold_local

from hmc_mir.align.isa_dtw import DTW_Cost_To_AccumCostAndSteps, DTW_GetPath

### SA_BCQT ###

def binarize_cqt(cqt):
    """Uses a local threshold for each frequency bin to binarize the input CQT.
    
    Args:
        cqt (np.ndarray): The CQT to be binarized
    
    Returns:
        binarized (np.ndarray): The binarized CQT
    """
    rows = cqt.shape[0]
    bin_size = 12
    context = 6
    binarized = []
    for i in range(0, rows, bin_size):
        if i - context < 0:
            data = cqt[:i + context]
        elif i + context >= rows:
            data = cqt[i - context:]
        else:
            data = cqt[i-context: i+context+bin_size]
        thresh = threshold_triangle(data)
        frequency_bin = cqt[i: i+bin_size]
        x1 = frequency_bin > thresh
        binarized.extend(x1)
    return np.array(binarized).astype(float)

@njit(parallel = True)
def calculate_cost_fast(query, ref):
    m, n1 = query.shape
    m, n2 = ref.shape
    result = np.zeros((n1, n2))
    for j1 in prange(n1):
        for j2 in prange(n2):
            for i in prange(m):
                result[j1, j2] += query[i, j1] * ref[i, j2]
    return result

def calculate_cost(query, ref):
    """Calculates the negative normalized cost between the query and reference.
    
    Args:
        query (np.ndarray): The binarized CQT of the part
        ref (np.ndarray): The binarized CQT of the full mix

    Returns:
        result (np.ndarray): The negative normalized cost matrix
    """
    cost = calculate_cost_fast(query, ref)
    row_sums = query.sum(axis = 0) * -1
    result = cost / row_sums[:, None]
    result[result == np.inf] = 0
    result = np.nan_to_num(result)
    return result

def align_binarized_cqts(query, ref, steps = [1,1,1,2,2,1], weights = [1,1,2]):
    """Uses subsequence DTW and the negative normalized inner product to compute an alignment between the part and full mix.
    
    Args:
        query (np.ndarray): The binarized CQT of the part
        ref (np.ndarray): The binarized CQT of the full mix
        steps (list): The steps to be used in the subsequence DTW
        weights (list): The weights to be used in the subsequence DTW
    """
    # set params
    assert len(steps) % 2 == 0, "The length of steps must be even."
    dn = np.array(steps[::2], dtype=np.uint32)
    dm = np.array(steps[1::2], dtype=np.uint32)
    dw = weights
    subsequence = True
    parameter = {'dn': dn, 'dm': dm, 'dw': dw, 'SubSequence': subsequence}

    # Compute cost matrix
    cost = calculate_cost(query, ref)

    # DTW
    [D, s] = DTW_Cost_To_AccumCostAndSteps(cost, parameter)
    [wp, endCol, endCost] = DTW_GetPath(D, s, parameter)

    # Reformat the output
    wp = wp.T[::-1]
    return wp

def time_stretch_part(query, ref, alignment):
    """Uses the alignment computed from DTW to time stretch the query to have the same dimensions as the reference.
    
    Args:
        query (np.ndarray): The CQT of the part
        ref (np.ndarray): The CQT of the full mix
        alignment (np.ndarray): The alignment between the part and full mix
    
    Returns:
        feature_stretch (np.ndarray): The time stretched part
    """
    m, n = ref.shape
    feature_stretch = np.zeros((m, n))
    used = set(alignment[:, 1])
    for query_idx, ref_idx in alignment:
        feature_stretch[:, ref_idx] = query[:, query_idx]
    for j in range(n):
        if j not in used:
            feature_stretch[:, j] = feature_stretch[:, j-1]
    return feature_stretch

def stretch_segments(segments, wp):
    """Uses the alignment created from DTW to also time stretch the nonsilence segments accordingly.
    
    Args:
        segments (list): The nonsilence segments
        wp (np.ndarray): The alignment between the part and full mix
    
    Returns:
        segments (list): The time stretched nonsilence segments
    """
    wp = np.array(sorted(wp, key = lambda x: x[0]))
    query_preds = wp[:, 0]
    ref_preds = wp[:, 1]
    query_to_ref = np.interp(list(range(max(query_preds[-1], ref_preds[-1]) + 1)), query_preds, ref_preds)
    n = len(query_to_ref) - 1
    segments[-1][1] = min(segments[-1][1], n)
    return [[int(query_to_ref[a]), int(query_to_ref[b])] for (a, b) in segments]

def weight_segments(segments, part_cqt, fullmix_cqt):
    alphas = np.concatenate([np.linspace(0.1, 1.0, num = 20), np.arange(1, 11, 0.3), np.arange(10, 510, 10)])
    for segment in segments:
        part_segment = part_cqt[:, segment[0]: segment[1] + 1]
        fullmix_segment = fullmix_cqt[:, segment[0]: segment[1] + 1]
        assert part_segment.shape == fullmix_segment.shape
        best = float('-inf')
        result = 0
        for alpha in alphas:
            val = np.sum(np.minimum(part_segment*alpha, fullmix_segment) - np.maximum(part_segment*alpha - fullmix_segment, 0))
            if val > best:
                best = val
                result = alpha
        part_cqt[:, segment[0]: segment[1] + 1] *= result

@njit(parallel = True)
def subtract_part(stretched_cqt, fullmix_cqt):
    """Subtracts the part CQT from the fullmix CQT elementwise."""
    m, n = stretched_cqt.shape
    
    for i in prange(m):
        for j in prange(n):
            fullmix_cqt[i, j] -= stretched_cqt[i, j]
            fullmix_cqt[i, j] = max(fullmix_cqt[i, j], 0)
    return fullmix_cqt

def isa_bcqt(part_cqt, fullmix_cqt, segments = []):
    """Performs the subtractive alignment algorithm between the part CQT and the full mix CQT

    First aligns the binarized CQTs, then uses the alignment to: time-stretch the part CQT, then perform reweighting, and then subtract the part CQT from the full mix CQT.

    Args:
        part_cqt (np.ndarray): The CQT of the part that is to be aligned/subtracted
        fullmix_cqt (np.ndarray): The CQT of the full mix
    
    Returns:
        fullmix_cqt (np.ndarray): The CQT of the full mix with the part subtracted
        wp (np.ndarray): The warping path
    """

    part_binarized, fullmix_binarized = binarize_cqt(part_cqt), binarize_cqt(fullmix_cqt)
    wp = align_binarized_cqts(part_binarized, fullmix_binarized)
    stretched_part = time_stretch_part(part_cqt, fullmix_cqt, wp)
    if segments:
        stretched_segments = stretch_segments(segments, wp)
        weight_segments(stretched_segments, stretched_part, fullmix_cqt)
    subtract_part(stretched_part, fullmix_cqt)
    return fullmix_cqt, wp