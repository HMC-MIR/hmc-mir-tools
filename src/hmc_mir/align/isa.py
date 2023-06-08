'''Iterative Subtractive Alignment

Code is from https://github.com/HMC-MIR/PianoTrioAlignment and https://archives.ismir.net/ismir2021/paper/000101.pdf
'''
from numba import njit, prange
import numpy as np
from skimage.filters import threshold_triangle
from scipy.spatial.distance import cdist
import librosa as lb
from sklearn import mixture

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
            data = cqt[:i + context + bin_size]
        elif i + context >= rows:
            data = cqt[i - context:]
        else:
            data = cqt[i-context: i+context+bin_size]
        thresh = threshold_triangle(data)
        frequency_bin = cqt[i: i+bin_size]
        x1 = frequency_bin > thresh
        binarized.extend(x1)
    return np.array(binarized).astype(float)

def calculate_binary_cost(query, ref):
    """Calculates the negative normalized cost between the query and reference.
    
    Args:
        query (np.ndarray): The binarized CQT of the part
        ref (np.ndarray): The binarized CQT of the full mix

    Returns:
        result (np.ndarray): The negative normalized cost matrix
    """
    cost = np.inner(query.T, ref.T)
    row_sums = query.sum(axis = 0) * -1
    result = cost / row_sums[:, None]
    result[result == np.inf] = 0
    result = np.nan_to_num(result)
    return result

@njit(parallel = True)
def calculate_neg_norm_min_cost(query, ref):
    """Calculates the negative normalized min cost between the query and reference."""
    m, n1 = query.shape
    m, n2 = ref.shape
    result = np.zeros((n1, n2))
    col_sums = np.zeros(n1)
    for j1 in prange(n1):
        for j2 in prange(n2):
            for i in prange(m):
                result[j1, j2] += min(query[i, j1], ref[i, j2])
    
    for j1 in prange(n1):
        for i in prange(m):
            col_sums[j1] += query[i, j1]

    for j1 in prange(n1):
        for j2 in prange(n2):
            result[j1, j2] *= -1
            result[j1, j2] /= col_sums[j1]
    return result

def calculate_cosine_cost(X,Y):
    cost = cdist(X,Y,'cosine')
    cost[np.isnan(cost)] = 0
    return cost

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
    """Uses the alignment created from DTW to weight the nonsilence segments accordingly.

    Args:
        segments (list): The nonsilence segments
        part_cqt (np.ndarray): The CQT of the part
        fullmix_cqt (np.ndarray): The CQT of the full mix
    
    Returns:
        segments (list): The weighted nonsilence segments
    """
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
    """Subtracts the part CQT from the fullmix CQT elementwise.

    Args:
        stretched_cqt (np.ndarray): The time stretched part CQT
        fullmix_cqt (np.ndarray): The CQT of the full mix
    
    Returns:
        fullmix_cqt (np.ndarray): The CQT of the full mix with the part CQT subtracted
    """
    m, n = stretched_cqt.shape
    
    for i in prange(m):
        for j in prange(n):
            fullmix_cqt[i, j] -= stretched_cqt[i, j]
            fullmix_cqt[i, j] = max(fullmix_cqt[i, j], 0)
    return fullmix_cqt

def calculate_cqt(audio, sr = 22050, hop_length = 512, bins = 12):
    """Calculates the CQT of the audio.

    Args:
        audio (np.ndarray): The audio to be converted to CQT
        sr (int): The sampling rate of the audio
        hop_length (int): The hop length of the CQT
        bins (int): The number of bins per octave

    Returns:
        cqt (np.ndarray): The CQT of the audio
    """
    return np.abs(lb.core.cqt(audio, n_bins = 8 * bins, bins_per_octave = bins, sr=sr, hop_length=hop_length))

def cqt_to_chroma(cqt):
    """Converts a CQT to a chroma representation.
    
    Args:
        cqt (np.ndarray): The CQT to be converted to chroma
    
    Returns:
        chroma (np.ndarray): The chroma representation of the CQT
    """
    chroma_map = lb.filters.cq_to_chroma(cqt.shape[0])
    chromagram = chroma_map.dot(cqt)
    chromagram = lb.util.normalize(chromagram, norm = 2)
    return chromagram

def align_subsequence_dtw(cost, steps = [1, 1, 1, 2, 2, 1], weights = [1, 1, 2]):
    """Uses subsequence DTW and the negative normalized cost metric to compute an alignment between
       the part and full mix."""
    assert len(steps) % 2 == 0, "The length of steps must be even."
    dn = np.array(steps[::2], dtype=np.uint32)
    dm = np.array(steps[1::2], dtype=np.uint32)
    dw = weights
    subsequence = True
    parameter = {'dn': dn, 'dm': dm, 'dw': dw, 'SubSequence': subsequence}
    
    # DTW
    [D, s] = DTW_Cost_To_AccumCostAndSteps(cost, parameter)
    [wp, endCol, endCost] = DTW_GetPath(D, s, parameter)

    # Reformat the output
    wp = wp.T[::-1]
    return wp


def frame_to_time(frame, hop_length = 512, sr = 22050):
    """ Converts a frame index to a time in seconds.

    Args:
        frame (int): The frame index
        hop_length (int): The hop length of the CQT
        sr (int): The sampling rate of the audio
    
    Returns:
        time (float): The time in seconds
    """
    return frame * hop_length / sr

def get_silence_intervals(silence_indices):
    """Uses a hard silence detection approach to identify contiguous regions of nonsilence.
    
    Args:
        silence_indices (list): A list of indices of silence
    
    Returns:
        silence_intervals (list): A list of intervals of nonsilence
    """
    cur_interval = []
    start = silence_indices[0]
    for i in range(len(silence_indices) - 1):
        if silence_indices[i] + 1 != silence_indices[i+1]:
            cur_interval.append((start, silence_indices[i]))
            start = silence_indices[i+1]
    cur_interval.append((start, silence_indices[-1]))
    silence_intervals = []
    for start, end in cur_interval:
        start_time = frame_to_time(start)
        end_time = frame_to_time(end)
        if end_time - start_time < 2:
            continue
        silence_intervals.append([start, end])
    return silence_intervals

def get_threshold(total_energies):
    """Uses a Gaussian mixture model fitted to an array of total energies to generate a threshold for nonsilence.
    
    Args:
        total_energies (np.ndarray): The total energies of the audio
    
    Returns:
        threshold (float): The threshold for nonsilence
    """
    model = mixture.GaussianMixture(n_components=3, covariance_type="full")
    model.fit(total_energies)
    pi, mu, sigma = model.weights_.flatten(), model.means_.flatten(), np.sqrt(model.covariances_.flatten())
    max_idx = np.argmax(mu)
    threshold = mu[max_idx] - 4 * sigma[max_idx]
    return threshold

def get_segments(audio, hop_length=512, N=2048,):
    """Given a piece of audio, calculates all the nonsilence segments within the audio using a hard silence detection approach with GMMs.
    
    Args:
        audio (np.ndarray): The audio to be segmented
        hop_length (int): The hop length of the STFT
        N (int): The window size of the STFT
    
    Returns:
        segments (list): A list of the nonsilence segments
    """
    stft = lb.stft(audio, n_fft=N, hop_length=hop_length)
    energies = np.sum(np.square(abs(stft)), axis=0)
    L = 32
    total_energies = []
    for i in range(len(energies)-L):
        total_energies.append(sum(energies[i:i+L]))
        
    total_energies = np.log(total_energies).reshape(-1, 1)
    threshold = get_threshold(total_energies)
    
    is_silence = [False] * (L//2 - 1)
    for energy in total_energies:
        if energy <= threshold:
            is_silence.append(True)
        else:
            is_silence.append(False)
    is_silence.extend([False] * (L//2))
    silence_indices = np.where(np.array(is_silence) == True)[0]
    silence_intervals = get_silence_intervals(silence_indices)
    nonsilence_segments = []
    cur = 0
    for start, end in silence_intervals:
        nonsilence_segments.append([cur, start])
        cur = end + 1
    nonsilence_segments.append([cur, len(is_silence)])
    return nonsilence_segments

def parse_wp(wp):
    """Parses the output of the alignment algorithm into a more readable format.

    Args:
        wp (np.ndarray): The output of the alignment algorithm
    
    Returns:
        wp (np.ndarray): The parsed output of the alignment algorithm
    """

    wp = np.array(sorted(wp, key = lambda x: x[0]))
    query_preds = wp[:, 0]
    ref_preds = wp[:, 1]
    return np.vstack((query_preds, ref_preds))

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
    cost = calculate_binary_cost(part_binarized, fullmix_binarized)
    wp = align_subsequence_dtw(cost)
    stretched_part = time_stretch_part(part_cqt, fullmix_cqt, wp)
    if segments:
        stretched_segments = stretch_segments(segments, wp)
        weight_segments(stretched_segments, stretched_part, fullmix_cqt)
    subtract_part(stretched_part, fullmix_cqt)
    return fullmix_cqt, parse_wp(wp)

def isa_cqt(part_cqt, fullmix_cqt, segments = []):
    """Performs the subtractive alignment algorithm between the part CQT and the full mix CQT by first time stretching the part CQT, then performing reweighting, and then subtracting the part CQT from the full mix CQT.
    
    Args:
        part_cqt (np.ndarray): The CQT of the part that is to be aligned/subtracted
        fullmix_cqt (np.ndarray): The CQT of the full mix
        segments (list): A list of the nonsilence segments
    
    Returns:
        fullmix_cqt (np.ndarray): The CQT of the full mix with the part subtracted
        wp (np.ndarray): The warping path
    """
    # part_cqt_with_noise = part_cqt + np.abs(np.random.randn(*part_cqt.shape)) * 1e-8
    # part_cqt_norm = part_cqt_with_noise / np.linalg.norm(part_cqt_with_noise, axis=0)
    
    # fullmix_cqt_with_noise = fullmix_cqt + np.abs(np.random.randn(*fullmix_cqt.shape)) * 1e-8
    # fullmix_cqt_norm = fullmix_cqt_with_noise / np.linalg.norm(fullmix_cqt_with_noise, axis=0)
    cost = calculate_neg_norm_min_cost(part_cqt, fullmix_cqt)
    wp = align_subsequence_dtw(cost)
    stretched_part = time_stretch_part(part_cqt, fullmix_cqt, wp)
    if segments:
        stretched_segments = stretch_segments(segments, wp)
        weight_segments(stretched_segments, stretched_part, fullmix_cqt)
    subtract_part(stretched_part, fullmix_cqt)
    return fullmix_cqt, parse_wp(wp)

def isa_chroma(part_cqt, fullmix_cqt, segments = []):
    """Performs the subtractive alignment algorithm between the part CQT and the full mix CQT

    First computes the chroma features, then aligns the chroma features, then uses the alignment to:
    time-stretch the part CQT, then perform reweighting, and then subtract the part CQT from the full mix CQT.
    
    Args:
        part_cqt (np.ndarray): The CQT of the part that is to be aligned/subtracted
        fullmix_cqt (np.ndarray): The CQT of the full mix
        segments (list): A list of the nonsilence segments
    
    Returns:
        fullmix_cqt (np.ndarray): The CQT of the full mix with the part subtracted
    """
    part_chroma, fullmix_chroma = cqt_to_chroma(part_cqt), cqt_to_chroma(fullmix_cqt)
    cost = calculate_cosine_cost(part_chroma.T, fullmix_chroma.T)
    wp = align_subsequence_dtw(cost)
    stretched_part = time_stretch_part(part_cqt, fullmix_cqt, wp)
    if segments:
        stretched_segments = stretch_segments(segments, wp)
        weight_segments(stretched_segments, stretched_part, fullmix_cqt)
    subtract_part(stretched_part, fullmix_cqt)
    return fullmix_cqt, parse_wp(wp)
