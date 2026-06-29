"""
archiver_core.py

A physical, data-driven screen recording archival pipeline.

================================================================================
PHYSICAL & CODING-THEORETIC ARCHITECTURE:
================================================================================
Every visual and temporal encoding parameter in this pipeline is dynamically
modeled from the statistical properties of the active audiovisual segments,
complying strictly with the H.265 specification and x265 parameter limits.

1. Reference Picture Buffer (ref):
   - Bounded at a maximum of 6 references. In standard HEVC main profile, the DPB
     is capped at 8 references. Because B-pyramid is enabled by default,
     specification limits require capping L0 references to 6 to guarantee
     hardware decoder playback safety.

2. B-Frame Slicetype Decisions (bframe-bias):
   - Clipped to [-90, 100] to match x265 CLI limits and prevent driver warnings.

3. Loop Filter Deblocking Offsets (deblock):
   - Fully utilizes the physical loop filter range of [-6, 6] to balance
     vector text sharpness against noisy gradient smoothing.

4. Spatial AQ Strength (aq-strength):
   - Mapped across [0.0, 3.0] to bypass high-frequency noise regions while
     protecting clean backdrops from banding.
"""

import av
import numpy as np
import subprocess
import sys
import os
from tqdm import tqdm
import io
import bisect
from concurrent.futures import ProcessPoolExecutor

# ==============================================================================
# --- CONFIGURATION & PHYSICAL CONSTANTS ---
# ==============================================================================

# Time seeking parameters
START_TIME = 0.0            # Default start time. Can be overridden via command-line arguments.

AUDIO_BITRATE = "48k"       # Target bitrate for the compressed mono Opus audio track
SILENCE_DURATION = 1.0      # Minimum duration (seconds) to classify a segment as silent
MARGIN_SECS = SILENCE_DURATION / 2  # Padding applied to speech boundaries to prevent clipping

# Human speech band limits (Hz) used to calculate the spectral energy ratio
SPEECH_BAND_MIN = 300
SPEECH_BAND_MAX = 4000

# --- Video Geometry and Physics Constants ---
BIT_DEPTH = 8
MAX_PIXEL_DIFF = (2 ** BIT_DEPTH) - 1  # 255 (Max absolute difference between two 8-bit pixels)
BLOCK_DIM = 8                          # Dimension of the square pixel blocks (8x8)
BLOCK_SIZE = BLOCK_DIM * BLOCK_DIM    # 64 (Total pixels per block)

# Maximum possible sum of absolute differences in an 8x8 block (255 * 64)
MAX_BLOCK_SUM = MAX_PIXEL_DIFF * BLOCK_SIZE  # 16320

# Number of bins needed to represent every possible integer block sum [0, 16320]
HIST_SIZE = MAX_BLOCK_SUM + 1  # 16321

# Absolute physical noise floor of an 8-bit digital system.
# BLOCK_SIZE (64) represents an average absolute change of exactly 1.0 luma step per pixel.
# Any change below this is mathematically sub-quantization dither noise.
MIN_PHYSICAL_LO = BLOCK_SIZE
MAX_PHYSICAL_HI = BLOCK_SIZE * MAX_PIXEL_DIFF

# 255 discrete thresholds corresponding to average pixel differences of 1.0 to 255.0 steps
THRESHOLDS = np.arange(1, 256) * BLOCK_SIZE
EPSILON = np.finfo(np.float64).eps  # Machine epsilon used to prevent division-by-zero errors
BFRAMES_LIMIT = 16                  # Physical upper limit of consecutive B-frames
MAX_RC_LOOKAHEAD = 250              # Upper limit for x265 rate control lookahead buffer

# --- Analytical Algorithmic Complexity Crossover ---
# Derived from first-order complexity theory comparing operational densities:
# Broadcast (N * T) vs. Suffix-Sum Bincount (N + M).
# Crossover occurs exactly where N * T = N + M => N = M / (T - 1)
CROSSOVER_LIMIT = HIST_SIZE // (THRESHOLDS.size - 1)  # Exactly 64 blocks


def safe_float(val):
    """
    Ensures float values are converted to string format using '.' as the decimal
    separator, bypassing any system locale configurations (e.g., European ',' commas)
    that would break FFmpeg's command-line parser.
    """
    return format(float(val), 'f')


def extract_luma_from_ndarray(arr, height, width):
    """
    Extracts the Y (Luma) channel as a 2D ndarray of shape (height, width)
    directly from the decoded frame array to avoid redundant memory copies.
    """
    if arr.ndim == 2:
        # Planar YUV format (e.g., yuv420p, yuv422p).
        # The Y plane occupies the top 'height' rows.
        return arr[:height, :].astype(np.int16)
    elif arr.ndim == 3:
        if arr.shape[0] == 3:
            # Planar format with channel-first ordering (e.g., yuv444p, gbrp).
            # The first channel represents Y or G.
            return arr[0, :, :].astype(np.int16)
        elif arr.shape[2] == 3:
            # Interleaved packed format (e.g., rgb24, bgr24).
            # Apply standard ITU-R BT.601 luma coefficients.
            return (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.int16)
        elif arr.shape[2] == 1:
            # Grayscale format
            return arr[:, :, 0].astype(np.int16)
    # Default fallback slice
    return arr[:height, :width].astype(np.int16)


# ==============================================================================
# --- OUTPUT MULTIPLEXING (LOGGING) --------
# ==============================================================================

class Tee:
    """
    Duplicates output writes to both the standard console stream and a log file.
    Filters raw progression streams to avoid writing incremental tqdm bar changes to the file.
    """
    def __init__(self, original_stream, file_stream, is_stderr=False):
        self.original_stream = original_stream
        self.file_stream = file_stream
        self.is_stderr = is_stderr

    def write(self, data):
        self.original_stream.write(data)
        if self.file_stream:
            if self.is_stderr and '\r' in data and '\n' not in data:
                return
            self.file_stream.write(data)

    def flush(self):
        self.original_stream.flush()
        if self.file_stream:
            self.file_stream.flush()

    def __getattr__(self, attr):
        return getattr(self.original_stream, attr)


# ==============================================================================
# --- MATHEMATICAL CLASS SEPARATION --------
# ==============================================================================

def perform_otsu_sweep(data, weights=None):
    """
    Class 3 Otsu Thresholding Sweep.

    Splits a 1D probability distribution into three distinct, mathematically
    optimal classes by maximizing the between-class variance (minimizing intra-class variance).

    This classification is a cornerstone of the screen archiving pipeline:
      - Class 0 (Background Noise): Contains sub-pixel luma fluctuations, sensor/dither noise.
      - Class 1 (Micro-Changes): Represents cursor blinking, pulsing microphone indicators, UI hover states.
      - Class 2 (Macro-Changes): Represents critical visual updates like typing new text, drawing lines, or slide transitions.
    """
    if data.size == 0:
        return 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    d_min, d_max = np.min(data), np.max(data)
    if np.isclose(d_min, d_max, atol=EPSILON):
        return float(d_min), float(d_max), 0.0, (float(d_min), float(d_min), float(d_min)), (1.0, 0.0, 0.0)

    # Bin the data deterministically into 256 intervals
    hist, bin_edges = np.histogram(data, bins=256, range=(d_min, d_max), weights=weights)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total_samples = np.sum(hist)
    if total_samples < EPSILON:
        return float(d_min), float(d_max), 0.0, (float(d_min), float(d_min), float(d_min)), (1.0, 0.0, 0.0)

    # Normalize histogram to get the probability density function (PDF)
    prob = hist.astype(np.float64) / total_samples

    # Compute cumulative probability (omega) and cumulative first-moment (mu)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_centers)
    mu_total = mu[-1]

    # Calculate total variance of the entire dataset
    total_var = np.sum(prob * (bin_centers - mu_total) ** 2)

    max_var_b = -1.0
    best_t1, best_t2 = 0, 0

    # Exhaustive search over all possible dual-threshold pairs (t1, t2)
    for t1 in range(len(prob) - 2):
        w0 = omega[t1]
        if w0 < EPSILON: continue
        m0 = mu[t1] / w0  # Mean of Class 0

        for t2 in range(t1 + 1, len(prob) - 1):
            w1 = omega[t2] - w0      # Weight of Class 1
            w2 = 1.0 - omega[t2]     # Weight of Class 2

            if w1 < EPSILON or w2 < EPSILON: continue

            m1 = (mu[t2] - mu[t1]) / w1       # Mean of Class 1
            m2 = (mu_total - mu[t2]) / w2     # Mean of Class 2

            # Calculate between-class variance
            var_b = (w0 * (m0 ** 2) + w1 * (m1 ** 2) + w2 * (m2 ** 2)) - (mu_total ** 2)

            # Maximize between-class variance
            if var_b > max_var_b:
                max_var_b, best_t1, best_t2 = var_b, t1, t2

    # Separability represents the proportion of variance explained by the 3-class model
    separability = max_var_b / total_var if total_var > EPSILON else 0.0

    w0_f = omega[best_t1]
    w1_f = omega[best_t2] - w0_f
    w2_f = 1.0 - omega[best_t2]

    m_classes = (
        (mu[best_t1] / w0_f) if w0_f > EPSILON else bin_centers[0],
        ((mu[best_t2] - mu[best_t1]) / w1_f) if w1_f > EPSILON else bin_centers[best_t1],
        ((mu_total - mu[best_t2]) / w2_f) if w2_f > EPSILON else bin_centers[best_t2]
    )

    omega_classes = (w0_f, w1_f, w2_f)

    return bin_edges[best_t1], bin_edges[best_t2], separability, m_classes, omega_classes


# ==============================================================================
# --- PHASE 1: AUDIO PROFILE COLLECTION ----
# ==============================================================================

def collect_audio_stats(input_file):
    """
    Decodes and profiles the audio stream to extract root-mean-square (RMS) energy,
    zero-crossing rate (ZCR), and spectral energy ratios in the human speech band.
    """
    container = av.open(input_file)
    a_stream = container.streams.audio[0]
    sr = a_stream.rate
    dur = float(container.duration / av.time_base) if container.duration else a_stream.duration * a_stream.time_base

    start_sec = float(START_TIME) if START_TIME else 0.0
    if start_sec > 0:
        container.seek(int(start_sec / a_stream.time_base), stream=a_stream)

    print("Phase 1/4: Collecting Raw Audio Data...")
    audio_stats = []

    tqdm_total = max(0.0, dur - start_sec)
    with tqdm(total=tqdm_total, unit="sec", desc="Analyzing Audio") as pbar:
        for frame in container.decode(a_stream):
            ts = float(frame.pts * a_stream.time_base) if frame.pts is not None else 0.0
            if ts < start_sec:
                continue

            raw = frame.to_ndarray()

            samples = raw.astype(np.float64).mean(axis=0).flatten() if raw.ndim > 1 else raw.flatten().astype(np.float64)

            # Normalize integer PCM samples to float [-1.0, 1.0] using np.float64 bounds
            if np.issubdtype(raw.dtype, np.integer):
                samples /= np.float64(np.iinfo(raw.dtype).max)

            if samples.size < 2:
                pbar.update(frame.samples / sr)
                continue

            # Time-domain metrics
            rms = np.sqrt(np.mean(samples ** 2))
            zcr = np.mean(np.abs(np.diff(np.sign(samples)))) / 2

            # Frequency-domain metrics (Real FFT)
            fft_data = np.abs(np.fft.rfft(samples))
            freqs = np.fft.rfftfreq(len(samples), 1 / sr)
            total_e = np.sum(fft_data)

            # Calculate the proportion of spectral energy residing in the human speech band
            fft_ratio = np.sum(fft_data[(freqs >= SPEECH_BAND_MIN) & (freqs <= SPEECH_BAND_MAX)]) / total_e if total_e > 0 else 0

            audio_stats.append({'ts': ts, 'rms': rms, 'zcr': zcr, 'fft': fft_ratio})
            pbar.update(frame.samples / sr)

    container.close()
    return audio_stats, dur


# ==============================================================================
# --- PHASE 2: INTERVAL CALCULATION --------
# ==============================================================================

def determine_intervals(audio_stats, dur, fps):
    """
    Generates a voice-driven timeline of active keep/drop segments.
    """
    print("Phase 2/4: Generating Speech-Driven Keep Intervals...")

    rms_vals = np.array([f['rms'] for f in audio_stats])
    zcr_vals = np.array([f['zcr'] for f in audio_stats])
    fft_vals = np.array([f['fft'] for f in audio_stats])

    # 1. Determine voice activity thresholds using Otsu sweeps
    non_zero_rms = rms_vals[rms_vals > 0]
    if len(non_zero_rms) == 0:
        auto_rms_thresh = 0.0
    else:
        min_audio_magnitude = np.min(non_zero_rms)
        safe_rms = np.clip(rms_vals, a_min=min_audio_magnitude, a_max=None)
        t1_rms, t2_rms, _, _, _ = perform_otsu_sweep(np.log10(safe_rms))
        auto_rms_thresh = 10 ** t2_rms  # Map back from logarithmic scale

    t1_zcr, t2_zcr, _, _, _ = perform_otsu_sweep(zcr_vals)
    auto_zcr_thresh = t1_zcr  # ZCR is lower for speech (vowels) than fricative noise

    t1_fft, t2_fft, _, _, _ = perform_otsu_sweep(fft_vals)
    auto_fft_thresh = t2_fft  # FFT energy ratio is high inside the speech band

    print(f"  > Audio Thresholds: RMS={auto_rms_thresh}, ZCR={auto_zcr_thresh}, FFT={auto_fft_thresh}")

    start_sec = float(START_TIME) if START_TIME else 0.0
    frame_dur = 1.0 / fps
    keep_intervals, current_start = [], None

    # 2. Audio Pass: Isolate all active voice segments
    for f in audio_stats:
        is_speech = (f['rms'] > auto_rms_thresh) and (f['zcr'] < auto_zcr_thresh) and (f['fft'] > auto_fft_thresh)
        if is_speech:
            if current_start is None: current_start = f['ts']
        elif current_start is not None:
            # Apply MARGIN_SECS padding to prevent conversational clipping
            keep_intervals.append((max(start_sec, current_start - MARGIN_SECS),
                                   min(dur, f['ts'] + MARGIN_SECS)))
            current_start = None
    if current_start is not None:
        keep_intervals.append((max(start_sec, current_start - MARGIN_SECS), dur))

    if not keep_intervals:
        return [(start_sec, start_sec + frame_dur)]

    # 3. Merge Pass: Sort and merge overlapping or adjacent intervals
    keep_intervals.sort()
    merged = [keep_intervals[0]]
    for next_s, next_e in keep_intervals[1:]:
        if next_s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], next_e))
        else:
            merged.append((next_s, next_e))

    return merged


# ==============================================================================
# --- PHASE 3: VIDEO PROFILING ENGINE ------
# ==============================================================================


def worker_analyze_sweep(args):
    """
    Parallel Worker Process.

    Decodes a temporal segment (chunk) of the video and simulates the block-level
    decimation behavior of FFmpeg's 'mpdecimate' filter using active feedback thresholds.
    """
    input_file, segments, filter_hi, filter_lo, filter_n = args
    frame_stats = []
    activity_log = []
    local_active_blocks = 0
    local_total_blocks = 0
    local_frames_processed = 0

    # Deterministic magnitude histogram containing counts of all block sums
    local_mag_hist = np.zeros(HIST_SIZE, dtype=np.int64)

    try:
        container = av.open(input_file)
        v_stream = container.streams.video[0]

        v_stream.codec_context.thread_count = 0  # 0 enables auto-detection of CPU cores
        v_stream.codec_context.thread_type = "AUTO"  # Let FFmpeg choose optimal multithreading

        for start_time, end_time in segments:
            # Seek to the keyframe nearest to the start_time of this specific sub-segment
            container.seek(int(start_time / float(v_stream.time_base)), stream=v_stream)
            ref_frame = None

            for frame in container.decode(v_stream):
                ts = float(frame.pts * v_stream.time_base) if frame.pts is not None else 0.0
                if ts < start_time:
                    continue
                if ts > end_time: break  # Cease decoding when exceeding chunk boundary
                local_frames_processed += 1

                # Access the raw Y (Luma) plane directly.
                plane = frame.planes[0]
                arr = np.frombuffer(plane, np.uint8).reshape(plane.height, plane.line_size)
                curr_frame = arr[:, :plane.width].astype(np.int16)

                if ref_frame is None:
                    ref_frame = curr_frame
                    continue

                # --- SPEED OPTIMIZATION: Fast-Fail ---
                diff = np.abs(curr_frame - ref_frame)
                if np.max(diff) == 0:
                    continue

                h, w = curr_frame.shape
                num_blocks = (h // 8) * (w // 8)
                local_total_blocks += num_blocks

                # Subdivide the frame difference matrix into 8x8 block matrices
                blocks = diff[:(h // 8) * 8, :(w // 8) * 8].reshape(h // 8, 8, w // 8, 8)
                # Sum the absolute differences inside each 8x8 block
                block_sums = blocks.sum(axis=(1, 3)).flatten()
                active_sums = block_sums[block_sums > 0]

                if active_sums.size > 0:
                    local_active_blocks += active_sums.size
                    activity_log.append((ts, np.max(active_sums)))

                    # OPTIMIZATION: Analytically Derived Hybrid Threshold Counter
                    # Bypasses arbitrary thresholds by utilizing the complexity-theory crossover limit (64 blocks).
                    if active_sums.size < CROSSOVER_LIMIT:
                        counts = np.sum(active_sums[:, None] > THRESHOLDS, axis=0)
                    else:
                        bincount = np.bincount(active_sums, minlength=HIST_SIZE)
                        suffix_sum = np.cumsum(bincount[::-1])[::-1]
                        suffix_sum_padded = np.append(suffix_sum, 0)
                        counts = suffix_sum_padded[THRESHOLDS + 1]

                    frame_stats.append(counts.astype(np.int32))

                    # Deterministically increment the global luma-magnitude histogram
                    np.add.at(local_mag_hist, active_sums, 1)

                    # Emulate mpdecimate's dual-gate frame keeping logic
                    # Opt: dead condition check removed. Iteration 1 starts naturally with lo=0, n=0
                    num_hi = np.sum(block_sums > filter_hi)
                    num_lo = np.sum(block_sums > filter_lo)
                    is_kept = (num_hi > 0) or (num_lo > filter_n)

                    # If the frame is kept by mpdecimate, we update the reference frame
                    if is_kept:
                        ref_frame = curr_frame

        container.close()
    except Exception as e:
        print(f"Worker error: {e}")

    return (np.array(frame_stats, dtype=np.int32) if frame_stats else np.empty((0, len(THRESHOLDS)), dtype=np.int32),
            local_mag_hist, local_active_blocks, local_total_blocks, local_frames_processed, activity_log)


def calculate_radiometric_constants_raw(counts_data, mag_hist, g_active, g_total):
    """
    Performs Otsu sweeps over the collected data arrays to find the optimal
    noise-separation thresholds, keeping them as high-precision floats.
    Now calculates the normalized Shannon Entropy of active changes.

    Note: Removed unused `g_frames` argument from function signature.
    """
    if counts_data.shape[0] == 0 or g_total == 0:
        return float(MAX_BLOCK_SUM), 100.0, 1.0, 0.0, 0.0, 0.0, 0.0

    # Calculate Shannon Entropy strictly on active differences (ignoring index 0)
    active_mag_hist = mag_hist[1:]
    total_active = np.sum(active_mag_hist)
    if total_active > 0:
        p_active = active_mag_hist[active_mag_hist > 0].astype(np.float64) / total_active
        shannon_entropy = -np.sum(p_active * np.log2(p_active))
        # Max entropy for 16320 possible non-zero values
        norm_entropy = shannon_entropy / np.log2(MAX_BLOCK_SUM)
    else:
        norm_entropy = 0.0

    non_zero_indices = np.flatnonzero(mag_hist)
    if non_zero_indices.size < 10:
        _, t2, separability, m_classes, omega_classes = 100.0, 1000.0, 0.5, (10.0, 50.0, 500.0), (0.9, 0.08, 0.02)
    else:
        indices = np.arange(HIST_SIZE)
        _, t2, separability, m_classes, omega_classes = perform_otsu_sweep(indices, weights=mag_hist)

    # Map LO to the higher Otsu threshold (t2)
    best_lo = max(float(MIN_PHYSICAL_LO), float(t2))

    # Physically lock HI to the maximum block sum (16320)
    best_hi = float(MAX_PHYSICAL_HI)

    # Calculate spatial-temporal entropy metrics
    p_nonzero = g_active / g_total

    # omega_metric tracks structural macro-change probability (Class 2)
    omega_metric = p_nonzero * omega_classes[2]
    phi_metric = 1.0 - (m_classes[0] / (m_classes[2] + EPSILON))

    # Match float threshold to closest threshold index
    lo_idx = np.argmin(np.abs(THRESHOLDS - best_lo))
    frame_activity_counts = counts_data[:, lo_idx]
    active_frame_counts = frame_activity_counts[frame_activity_counts > 0]

    if active_frame_counts.size < 10:
        n_blocks = 1.0
    else:
        # Keep block count threshold as a float
        n_blocks = float(perform_otsu_sweep(active_frame_counts)[0])

    return best_hi, best_lo, n_blocks, separability, omega_metric, phi_metric, norm_entropy


def profile_video_sweep(input_file, keep_intervals, width, height):
    """
    Core Search, Convergence, and Radiometric Extraction Engine.

    Runs visual analysis calibration sweeps directly on the video segments,
    tracks parameter derivatives to prevent oscillations, extrapolates infinite-limit
    fixed point constants, and extracts spatial-temporal metrics.
    """
    print("Phase 3/4: Profiling Video Blocks (Full Timeline Convergence via PyAV)...")

    # --- Fast Metadata-Only Scan for Keyframe Timestamps ---
    kf_times = []
    try:
        with av.open(input_file) as kf_container:
            v_stream_kf = kf_container.streams.video[0]
            # demux only yields packets, avoiding any video frame decoding
            for packet in kf_container.demux(v_stream_kf):
                # Ignore flushing / dummy packets that don't have timestamps
                if packet.pts is None:
                    continue
                if packet.is_keyframe:
                    # Map PTS to seconds using the stream's time base
                    kf_times.append(float(packet.pts * v_stream_kf.time_base))
    except Exception as e:
        print(f"  > Warning: Rapid keyframe pre-scan failed ({e}). Falling back to empty GOP map.")
        kf_times = []

    # Ensure times are sorted (some container types may occasionally yield slightly out-of-order DTS)
    kf_times.sort()

    total_blocks = (width // 8) * (height // 8)

    def extrapolate_infinite_limit(seq, min_val=0, max_val=MAX_BLOCK_SUM):
        """
        Extrapolates parameter patterns into their infinite-limit values using
        second-order auto-regression (AR(2)) with drift. Replaces the previous
        AR(1) model, which required monotonic convergence and broke on damped
        oscillation. AR(2) stability is checked via the roots of the characteristic
        polynomial — sequences that converge via damped oscillation are handled
        correctly without needing a separate oscillation-detection guard.
        """
        # AR(2) with intercept needs n-2 samples >= 3 unknowns (A1, A2, c),
        # so a minimum of 5 history points is required.
        if len(seq) < 5:
            return seq[-1], 0

        y = np.array(seq, dtype=np.float64)

        # Two-lag design matrix with intercept column: [y[n-1], y[n-2], 1] -> y[n]
        # Full AR(2) with drift: y[n] = A1*y[n-1] + A2*y[n-2] + c + eps
        X = np.column_stack([y[1:-1], y[:-2], np.ones(len(y) - 2)])
        Y = y[2:]

        # OLS via least-squares (robust to near-singular or exactly-determined cases)
        coeffs, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
        if rank < 3:
            return seq[-1], 0

        A1, A2, c = coeffs

        # AR(2) stability: both roots of z^2 - A1*z - A2 = 0 must lie inside the
        # unit circle. This covers monotonic convergence (real roots < 1) AND damped
        # oscillation (complex conjugate roots with |z| < 1), replacing the old
        # 0.0 < A < 1.0 guard which rejected all oscillating sequences outright.
        roots = np.roots([1, -A1, -A2])
        if not np.all(np.abs(roots) < 1.0):
            return seq[-1], 0

        # AR(2) fixed point: y* = c / (1 - A1 - A2)
        denom = 1.0 - A1 - A2
        if abs(denom) < 1e-8:
            return seq[-1], 0

        limit = c / denom
        if not (min_val <= limit <= max_val):
            return seq[-1], 0

        current_val = seq[-1]
        distance = abs(current_val - limit)

        # Step estimate uses the spectral radius (dominant root magnitude) as the
        # convergence rate — valid for both real and complex roots. Replaces log(A)
        # from the AR(1) model.
        spectral_radius = np.max(np.abs(roots))
        if distance > 0.5 and spectral_radius > EPSILON:
            steps = int(np.ceil(np.log(0.5 / distance) / np.log(spectral_radius)))
        else:
            steps = 0

        return limit, steps

    # 1. HARDWARE CONCURRENCY ASSESSMENT
    # Sub-linear worker scaling minimizes scheduler context-switch overhead
    logical_cores = os.cpu_count() or 1
    num_workers = max(1, int(np.round(np.sqrt(logical_cores))))

    # 2. INTERSECT TIMELINE (Speech-Interval Snapped GOPs)
    # Partitions keep-intervals using the physical keyframes that fall inside them.
    # This guarantees zero speech data loss while maintaining clean decodes.
    active_gop_segments = []
    for s, e in keep_intervals:
        idx_start = bisect.bisect_right(kf_times, s)
        idx_end = bisect.bisect_left(kf_times, e)
        internal_kfs = kf_times[idx_start:idx_end]

        boundary_pts = [s] + internal_kfs + [e]
        for i in range(len(boundary_pts) - 1):
            active_gop_segments.append((boundary_pts[i], boundary_pts[i + 1]))

    active_gop_durs = np.array([e - s for s, e in active_gop_segments])
    total_keep_dur = sum(active_gop_durs)

    # 3. DYNAMIC BOUNDS CALCULATION
    # - T_min: Prevents IPC/Process scheduling overhead from dominating runtimes (max 16 tasks per worker)
    # - T_max: Ensures balanced workload distribution (min 2 tasks per worker)
    T_min = total_keep_dur / (16.0 * num_workers)
    T_max = total_keep_dur / (2.0 * num_workers)

    # 4. LOGARITHMIC GOP OTSU SWEEP
    if active_gop_durs.size >= 10:
        log_gops = np.log10(np.clip(active_gop_durs, a_min=0.1, a_max=None))
        _, t2_log, _, _, _ = perform_otsu_sweep(log_gops)
        T_static = 10 ** t2_log
    else:
        # Mathematical fallback to the geometric mean of the hardware bounds
        T_static = np.sqrt(T_min * T_max)

    # Clamp the static threshold strictly within the hardware bounds
    T_static = np.clip(T_static, T_min, T_max)

    # 5. DYNAMIC CHUNKING (Stillness vs. Motion Classification)
    # - Static intervals (>= T_static) are emitted as isolated chunks to maximize skip probability.
    # - Active intervals (< T_static) are accumulated together into single chunks to reduce IPC overhead.
    groups = []
    current_group = []
    accumulated_dur = 0.0

    for seg in active_gop_segments:
        s_time, e_time = seg
        g_dur = e_time - s_time

        if g_dur >= T_static:
            if current_group:
                groups.append(current_group)
                current_group = []
                accumulated_dur = 0.0
            groups.append([seg])
        else:
            current_group.append(seg)
            accumulated_dur += g_dur
            if accumulated_dur >= T_static:
                groups.append(current_group)
                current_group = []
                accumulated_dur = 0.0

    if current_group:
        groups.append(current_group)

    tasks = [(input_file, g) for g in groups]
    avg_chunk_dur = total_keep_dur / len(groups) if groups else 0.0
    print(
        f"  > Divided keep intervals ({safe_float(total_keep_dur)}s active) into {len(groups)} hardware-scaled segments "
        f"({num_workers} workers, avg chunk: {safe_float(avg_chunk_dur)}s)...")

    lo_history = []
    n_history = []
    prev_extrapolated_lo = None
    prev_extrapolated_n = None
    hi = float(MAX_PHYSICAL_HI)
    lo, n_blocks = 0.0, 0.0
    max_iters = 16
    relaxation_damping = 2.0 / 3.0
    static_chunks = set()

    # Guard metric fallback definitions to protect against empty iteration sets
    last_iter_metrics = {
        'separability': 0.0,
        'omega_metric': 0.0,
        'phi_metric': 0.0,
        'norm_entropy': 0.0,
        'activity_log': [],
        'g_act': 0,
        'g_tot': 0,
        'g_frm': 0
    }

    for iter_idx in range(max_iters):
        active_tasks = []
        task_mapping = []

        for i, t in enumerate(tasks):
            if i in static_chunks:
                continue
            active_tasks.append((t[0], t[1], hi, lo, n_blocks))
            task_mapping.append(i)

        if not active_tasks:
            break

        f_stats = []
        global_mag_hist = np.zeros(HIST_SIZE, dtype=np.int64)
        g_act, g_tot, g_frm = 0, 0, 0
        global_activity_log = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for local_idx, (res_counts, res_hist, act_cnt, tot_cnt, frm_cnt, act_log) in enumerate(tqdm(
                    executor.map(worker_analyze_sweep, active_tasks),
                    total=len(active_tasks),
                    desc=f"Calibration Iter {iter_idx + 1}"
            )):
                original_idx = task_mapping[local_idx]
                if iter_idx == 0 and act_cnt == 0:
                    static_chunks.add(original_idx)

                if res_counts.size > 0:
                    f_stats.append(res_counts)
                global_mag_hist += res_hist
                g_act += act_cnt
                g_tot += tot_cnt
                g_frm += frm_cnt
                global_activity_log.extend(act_log)

        counts_data = np.vstack(f_stats) if f_stats else np.empty((0, len(THRESHOLDS)))

        # Retrieve new metrics including normalized entropy
        # Note: Unused g_frm (g_frames) removed from signature
        hi_new, lo_new, n_new, separability, omega_metric, phi_metric, norm_entropy_new = calculate_radiometric_constants_raw(
            counts_data, global_mag_hist, g_act, g_tot
        )

        last_iter_metrics = {
            'separability': separability,
            'omega_metric': omega_metric,
            'phi_metric': phi_metric,
            'norm_entropy': norm_entropy_new,  # Cache the calculated entropy
            'activity_log': global_activity_log,
            'g_act': g_act,
            'g_tot': g_tot,
            'g_frm': g_frm
        }

        # Apply relaxation damping and calculate deltas
        if iter_idx == 0:
            lo, n_blocks = lo_new, n_new
            lo_diff_abs, lo_diff_rel = 0.0, 0.0
            n_diff_abs, n_diff_rel = 0.0, 0.0
        else:
            lo_prev, n_prev = lo, n_blocks
            lo = relaxation_damping * lo + (1.0 - relaxation_damping) * lo_new
            n_blocks = relaxation_damping * n_blocks + (1.0 - relaxation_damping) * n_new

            lo_diff_abs = abs(lo - lo_prev)
            lo_diff_rel = (lo_diff_abs / lo_prev) * 100.0 if lo_prev > EPSILON else 0.0
            n_diff_abs = abs(n_blocks - n_prev)
            n_diff_rel = (n_diff_abs / n_prev) * 100.0 if n_prev > EPSILON else 0.0

        lo = max(float(MIN_PHYSICAL_LO), lo)
        lo_history.append(lo)
        n_history.append(n_blocks)

        # Generate structural log representation with absolute and relative delta shifts
        delta_str_lo = f" (Δ abs: {safe_float(lo_diff_abs)}, rel: {safe_float(lo_diff_rel)}%)" if iter_idx > 0 else " (Initial)"
        delta_str_n = f" (Δ abs: {safe_float(n_diff_abs)}, rel: {safe_float(n_diff_rel)}%)" if iter_idx > 0 else " (Initial)"

        print(
            f"    - Iteration {iter_idx + 1}:\n"
            f"      > Parameters: LO={safe_float(lo)}{delta_str_lo}\n"
            f"                    HI={safe_float(hi)} (Locked)\n"
            f"                    n={safe_float(n_blocks)}{delta_str_n}\n"
            f"      > State:      Sep={safe_float(separability)}, H_norm={safe_float(norm_entropy_new)}, "
            f"Omega={safe_float(omega_metric)}, Phi={safe_float(phi_metric)}"
        )

        # AR(2) Convergence Check
        # Oscillation detection is removed — AR(2) handles damped oscillation
        # natively via the characteristic-polynomial stability check. The sole
        # early-exit criterion is the extrapolated fixed point rounding to the
        # same integer on two consecutive iterations.
        if len(lo_history) >= 5:  # Minimum for AR(2) with intercept (3 unknowns)
            lo_est, lo_steps = extrapolate_infinite_limit(lo_history, min_val=MIN_PHYSICAL_LO)
            n_est, n_steps = extrapolate_infinite_limit(n_history, max_val=total_blocks)

            lo_est_rounded = int(round(lo_est))
            n_est_rounded = int(round(n_est))

            print(f"    > Current Projection L(n): LO={lo_est}≈{lo_est_rounded} (est. {lo_steps} iters) | "
                  f"n={n_est}≈{n_est_rounded} (est. {n_steps} iters)")

            if prev_extrapolated_lo is not None and prev_extrapolated_n is not None:
                if lo_est_rounded == prev_extrapolated_lo and n_est_rounded == prev_extrapolated_n:
                    print(
                        f"    - Extrapolated limit converged early at Iteration {iter_idx + 1} (LO={lo_est_rounded}, n={n_est_rounded}). Halting loop.")
                    break

            prev_extrapolated_lo = lo_est_rounded
            prev_extrapolated_n = n_est_rounded

    lo_extrapolated, _ = extrapolate_infinite_limit(lo_history, min_val=MIN_PHYSICAL_LO)
    n_blocks_extrapolated, _ = extrapolate_infinite_limit(n_history, max_val=total_blocks)

    best_lo = int(round(lo_extrapolated))
    best_hi = MAX_PHYSICAL_HI
    best_n = int(round(n_blocks_extrapolated))

    print(f"\n  > Global Sequence Extrapolation Successful.")
    print(f"  > Infinite-Limit Fixed Point: LO={best_lo}, HI={best_hi} (Locked), n={best_n}")

    separability = last_iter_metrics['separability']
    omega_metric = last_iter_metrics['omega_metric']
    phi_metric = last_iter_metrics['phi_metric']
    norm_entropy = last_iter_metrics['norm_entropy']  # Extract cached value
    activity_log = last_iter_metrics['activity_log']
    g_tot = last_iter_metrics['g_tot']
    g_frm = last_iter_metrics['g_frm']

    total_blocks_per_frame = (g_tot / g_frm) if g_frm > 0 else 1
    mpdecimate_frac = np.clip(best_n / total_blocks_per_frame, EPSILON, 1.0)

    print(f"\n--- Radiometric Optimization Results ---")
    print(
        f"  > Optimal HI={best_hi}, LO={best_lo}, n={best_n} (FRAC={safe_float(mpdecimate_frac)}), Sep={safe_float(separability)}")

    return best_hi, best_lo, mpdecimate_frac, separability, omega_metric, phi_metric, norm_entropy, activity_log


# ==============================================================================
# --- TEMPORAL & AUDIO EXPORT HELPERS ------
# ==============================================================================

def calculate_temporal_lookahead(merged_intervals, fps):
    """
    Calculates the optimal x265 rate control lookahead buffer based on the
    distribution of keeping-interval durations.
    """
    durations = np.array([(e - s) * fps for s, e in merged_intervals])
    if len(durations) < 3:
        return MAX_RC_LOOKAHEAD

    # Use Otsu to isolate the primary structural clusters
    t1_t, _, t_sep, _, _ = perform_otsu_sweep(durations)
    signal_mask = durations > t1_t
    if t_sep < EPSILON or not np.any(signal_mask):
        return MAX_RC_LOOKAHEAD

    # We set the lookahead equal to the mean duration of the active structural segments
    auto_lookahead_raw = np.mean(durations[signal_mask])
    return int(np.clip(round(auto_lookahead_raw), BFRAMES_LIMIT, MAX_RC_LOOKAHEAD))


def export_trimmed_audio(input_file, output_wav_path, merged_intervals, total_sec):
    """
    Extracts, resamples, and multiplexes the audio segments corresponding to the
    keeping intervals into a unified, clean mono PCM-S16LE temporary WAV file.
    """
    container = av.open(input_file)
    a_stream = container.streams.audio[0]
    sr = a_stream.rate

    start_sec = float(START_TIME) if START_TIME else 0.0
    if start_sec > 0:
        container.seek(int(start_sec / a_stream.time_base), stream=a_stream)

    abs_path = os.path.abspath(output_wav_path)
    out_container = av.open(abs_path, mode='w')
    out_s = out_container.add_stream('pcm_s16le', rate=sr)
    out_s.layout = 'mono'
    resampler = av.AudioResampler(format='s16', layout='mono', rate=sr)
    starts = [s for s, e in merged_intervals]

    with tqdm(total=total_sec + 1, unit="sec", desc="Muxing Audio") as pbar:
        for frame in container.decode(a_stream):
            ts = float(frame.pts * a_stream.time_base) if frame.pts is not None else 0.0
            if ts < start_sec:
                continue

            idx = bisect.bisect_right(starts, ts)
            if idx > 0 and ts <= merged_intervals[idx - 1][1]:
                frame.pts = None
                for out_frame in resampler.resample(frame):
                    for pkt in out_s.encode(out_frame): out_container.mux(pkt)
            pbar.update(frame.samples / sr)
        for out_frame in resampler.resample(None):
            for pkt in out_s.encode(out_frame): out_container.mux(pkt)
        for pkt in out_s.encode(None): out_container.mux(pkt)

    out_container.close()
    container.close()


# ==============================================================================
# --- PHASE 4: MASTER PIPELINE EXECUTION ---
# ==============================================================================

def run_pipeline():
    """
    Orchestrates the entire archival pipeline.

    1. Sets up the stream redirection to capture logs cleanly.
    2. Gathers general stream physics metadata (dimensions, frame rate).
    3. Checks for existing JSON parameter cache to skip analysis if desired.
    4. Compiles the voice-activity audio stats (or loads cached stats).
    5. Runs the full-timeline global-extrapolation video profiling sweep (or loads cached stats).
    6. Translates spatial-temporal metrics to x265 and mpdecimate variables.
    7. Generates keeping-intervals and exports the consolidated mono WAV file.
    8. Pipes the raw video frames corresponding to the keeping intervals directly
       to FFmpeg, executing a veryslow x265 encode with customized visual parameters.
    """
    global START_TIME
    if len(sys.argv) < 4:
        print("Error: Missing execution arguments.")
        print("Usage: python archiver_core.py <input_file> <output_file> <temp_audio_file> [start_time_seconds]")
        sys.exit(1)

    input_file, output_file, temp_audio_file = sys.argv[1], sys.argv[2], sys.argv[3]

    # Dynamically read START_TIME if passed as a 4th argument from PowerShell
    if len(sys.argv) >= 5:
        try:
            START_TIME = float(sys.argv[4])
            print(f"  > Dynamic Override: START_TIME set to {safe_float(START_TIME)} seconds.")
        except ValueError:
            print(f"Warning: Invalid start_time argument '{sys.argv[4]}'. Defaulting to {safe_float(START_TIME)}.")

    # --- Setup Logging Redirection ---
    log_file_path = output_file + ".log"
    cache_file_path = output_file + ".cache.json"

    try:
        log_file = open(log_file_path, 'w', encoding='utf-8')
    except Exception as e:
        print(f"Warning: Could not initiate log file at {log_file_path}: {e}")
        log_file = None

    # Save original references for clean stream restoration
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    if log_file:
        sys.stdout = Tee(original_stdout, log_file, is_stderr=False)
        sys.stderr = Tee(original_stderr, log_file, is_stderr=True)

    try:
        # --- Stream Metadata Pre-Scan ---
        # Fetch properties from input container metadata to avoid redundant initial opens.
        container = av.open(input_file)
        v_stream = container.streams.video[0]
        fps = float(v_stream.average_rate) if v_stream.average_rate else 10.0
        if fps <= 0:
            fps = 10.0
        width = v_stream.width
        height = v_stream.height
        pix_fmt = v_stream.pix_fmt
        container.close()

        # --- Dynamic Calibration Cache Verification ---
        import json
        use_cached = False
        cached_data = None

        if os.path.exists(cache_file_path):
            try:
                with open(cache_file_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)

                # Verify structural presence of essential metrics
                required_keys = ['merged', 'total_sec', 'best_hi', 'best_lo', 'mpdecimate_frac',
                                 'S_val', 'Omega', 'Phi', 'H_norm']
                if all(k in cached_data for k in required_keys):
                    response = input(
                        f"\n[CACHE] Found existing calibration cache at '{cache_file_path}'.\n"
                        f"Do you want to reuse these parameters and skip analysis? (y/n): "
                    ).strip().lower()
                    if response in ('y', 'yes'):
                        use_cached = True
                        print("[CACHE] Reusing cached analysis metrics. Skipping Phases 1-3.\n")
            except Exception as e:
                print(f"[CACHE] Warning: Failed to parse cache file ({e}). Re-analyzing stream.")

        if use_cached:
            # Unpack the cache file back into physical and spatial-temporal variables
            merged = [tuple(item) for item in cached_data['merged']]
            total_sec = cached_data['total_sec']
            best_hi = cached_data['best_hi']
            best_lo = cached_data['best_lo']
            mpdecimate_frac = cached_data['mpdecimate_frac']
            S_val = cached_data['S_val']
            Omega = cached_data['Omega']
            Phi = cached_data['Phi']
            H_norm = cached_data['H_norm']
        else:
            # --- Executing Phase 1 (Audio Stats) ---
            audio_stats, total_sec = collect_audio_stats(input_file)

            # --- Executing Phase 2 (Speech-Driven Keep Intervals) ---
            merged = determine_intervals(audio_stats, total_sec, fps)

            # --- Executing Phase 3 (Unified Video Profiling & Radiometric Calibration) ---
            best_hi, best_lo, mpdecimate_frac, S_val, Omega, Phi, H_norm, activity_log = profile_video_sweep(
                input_file, merged, width, height)

            # Write calibration metrics immediately to cache before beginning Master Encode
            try:
                cache_payload = {
                    'merged': merged,
                    'total_sec': total_sec,
                    'best_hi': best_hi,
                    'best_lo': best_lo,
                    'mpdecimate_frac': mpdecimate_frac,
                    'S_val': S_val,
                    'Omega': Omega,
                    'Phi': Phi,
                    'H_norm': H_norm
                }
                with open(cache_file_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_payload, f, indent=4)
                print(f"[CACHE] Successfully saved calibration metrics to: {cache_file_path}")
            except Exception as e:
                print(f"[CACHE] Warning: Could not write cache file: {e}")

        # --- Phase 4: Master Encode Preparation ---
        container = av.open(input_file)
        v_stream = container.streams.video[0]

        # Enable multithreaded decoding
        v_stream.codec_context.thread_count = 0
        v_stream.codec_context.thread_type = "AUTO"

        auto_lookahead = calculate_temporal_lookahead(merged, fps)
        print(f"  > Temporal Result: Lookahead={auto_lookahead}")

        export_trimmed_audio(input_file, temp_audio_file, merged, total_sec)

        resolution = f"{width}x{height}"
        input_pix_fmt = pix_fmt.replace('j', '')
        output_pix_fmt = input_pix_fmt

        try:
            canary_frame = next(container.decode(v_stream))
            bytes_per_frame = canary_frame.to_ndarray(format=pix_fmt).nbytes
            container.seek(0)
        except StopIteration:
            return

        io_buffer_size = bytes_per_frame * 64

        # ======================================================================
        # --- CODING-THEORETIC x265 PARAMETER MAPPINGS ---
        # ======================================================================

        # 1. DPB Limit / Reference Frames (ref)
        # Based on geometric decay of static blocks over scene transition rate Omega.
        # Bounded within 1 to 6 reference pictures (Strict H.265 specification ceiling
        # for standard-compliant decoders when B-pyramid remains active).
        auto_ref = int(np.clip(np.round(S_val / (Omega + EPSILON)), 1, 6))

        # 2. B-Frame Log-Odds Model (bframe-bias)
        # Models the symmetric log-odds (logit) of temporal stability vs. change.
        # Project log-odds via tanh to strictly match x265 bias bounds [-90, 100].
        b_stability_odds = (1.0 - Omega) / (Omega + EPSILON)
        b_bias_factor = np.tanh(np.log(b_stability_odds))
        auto_b_bias = int(np.clip(np.round(100.0 * b_bias_factor), -90, 100))

        # 3. Dynamic B-Frame Allocations (bframes)
        # Scales total consecutive B-frames from its ceiling of 16 down to a floor of 4,
        # based on combined change rate and entropy.
        auto_bframes = int(np.clip(np.round(BFRAMES_LIMIT * (1.0 - (Omega * H_norm))), 4, BFRAMES_LIMIT))

        # 4. State-Separability Scenecut Sensitivity (scenecut)
        # Bounded relative contrast sensitivity for frame change checks, mapped to [10, 90].
        auto_scenecut = int(np.clip(np.round(100.0 * S_val), 10, 90))

        # 5. Psychovisual Contrast Balance (psy-rd & psy-rdoq)
        # Preserves crisp vector edges while suppressing psychovisual spend on chaotic noise.
        psi = Phi * (1.0 - H_norm)
        auto_psy_rd = round(float(5.0 * psi), 2)
        auto_psy_rdoq = round(float(auto_psy_rd * 0.5), 2)

        # 6. Complementary Entropy Rate-Control (qcomp)
        # Curves rate distortion towards constant QP for text slides and constant rate for noise.
        # Clipped inside the standard x265 operational bounds [0.5, 0.95].
        auto_qcomp = round(float(np.clip(1.0 - H_norm, 0.5, 0.95)), 2)

        # 7. Balanced Deblocking Filter Offsets (deblock)
        # Linear projection of change entropy (smoothing need) vs visual contrast (sharpness need).
        # Fully maps onto the standard loop filter offset range of [-6, 6].
        deblock_balance = H_norm - Phi
        auto_deblock_offset = int(np.clip(np.round(6.0 * deblock_balance), -6, 6))
        # Note: Comma is used as separator inside x265-params to bypass colon parsing issues
        auto_deblock_str = f"{auto_deblock_offset},{auto_deblock_offset}"

        # 8. Contrast-Preserving Adaptive Quantization (aq-strength)
        # Protects smooth, flat vector backgrounds while bypassing random noise regions.
        # Fully maps onto the standard AQ operating bounds of [0.0, 3.0].
        auto_aq_strength = round(float(np.clip(3.0 * Phi * (1.0 - H_norm), 0.0, 3.0)), 2)

        # 9. Spatial-Temporal Edge Skip Tuning (tskip & rskip)
        # - tskip: Enables transform skip (lossless 4x4 bypass) for flat vector blocks if change entropy is low.
        # - rskip-edge: Early depth recursion skip percentage threshold. Mapped strictly inside standard [1, 10] bounds as an integer percentage.
        auto_tskip = 1 if H_norm < 0.35 else 0
        auto_rskip_edge = int(round(np.clip(1.0 + 9.0 * H_norm, 1.0, 100.0)))

        # --- Calculated Spatial-Temporal Metrics ---
        print(f"\n--- Calculated Spatial-Temporal Metrics ---")
        print(f"  > Visual Contrast Ratio (Phi):        {safe_float(Phi)}")
        print(f"  > Macro-Change Probability (Omega):   {safe_float(Omega)}")
        print(f"  > Otsu Separability (S_val):          {safe_float(S_val)}")
        print(f"  > Normalized Change Entropy (H_norm): {safe_float(H_norm)}")

        print(f"\n--- Physically Calibrated x265 Parameters ---")
        print(f"  > Max Reference Frames (ref):         {auto_ref}")
        print(f"  > Max Consecutive B-Frames:           {auto_bframes}")
        print(f"  > B-Frame Bias (Odds Ratio):          {auto_b_bias}")
        print(f"  > Scenecut Sensitivity:               {auto_scenecut}")
        print(f"  > Psychovisual RD strength:           {safe_float(auto_psy_rd)}")
        print(f"  > Psychovisual RDOQ trellis strength: {safe_float(auto_psy_rdoq)}")
        print(f"  > Rate Control qcomp (Curve Comp):    {safe_float(auto_qcomp)}")
        print(f"  > Deblock Filter Configuration:       deblock={auto_deblock_str}")
        print(f"  > AQ Strength (Bias toward flats):    {safe_float(auto_aq_strength)}")
        print(f"  > Transform Skip (Lossless 4x4):      tskip={auto_tskip}")
        print(f"  > Recursion Skip Edge Threshold:      rskip-edge-threshold={safe_float(auto_rskip_edge)}")
        print(f"----------------------------------------------------\n")

        # Configuration dict mapped strictly to the dynamic inputs
        # Note: Dict keys are logical categories; joined with colons to map into FFmpeg's -x265-params CLI flag
        x265_cfg = {
            "Profile": f"ref={auto_ref}",
            "Analysis": f"rd=6:rskip=2:rskip-edge-threshold={auto_rskip_edge}:rdoq-level=1:tu-intra-depth=4:tu-inter-depth=4:tskip={auto_tskip}",
            "Motion": "max-merge=5:subme=7:weightb=1:hme=1:hme-search=star,star,star:analyze-src-pics=1",
            "Intra": "strong-intra-smoothing=0:constrained-intra=1",  # Smooth off to keep vector text crisp
            "Psy": f"psy-rd={safe_float(auto_psy_rd)}:psy-rdoq={safe_float(auto_psy_rdoq)}",
            "GOP": f"open-gop=0:keyint=-1:min-keyint=0:scenecut={auto_scenecut}:hist-scenecut=1:rc-lookahead={auto_lookahead}:b-adapt=2:bframes={auto_bframes}:bframe-bias={auto_b_bias}:fades=1",
            "RC": f"crf=30:aq-mode=4:aq-strength={safe_float(auto_aq_strength)}:qp-adaptation-range=6.0:aq-motion=1:qg-size=8:qcomp={safe_float(auto_qcomp)}",
            "Filters": f"deblock={auto_deblock_str}:sao=0",  # SAO disabled to prevent ringing around letters
            "VUI": "repeat-headers=1:opt-qp-pps=1:opt-ref-list-length-pps=1:opt-cu-delta-qp=1"
        }

        print(f"\nPhase 4/4: Master Encode [x265 Veryslow 8-bit]")

        # Configure mpdecimate with our converged, infinite-limit parameters
        v_filter = f"mpdecimate=hi={safe_float(best_hi)}:lo={safe_float(best_lo)}:frac={safe_float(mpdecimate_frac)},setpts=PTS-STARTPTS"

        ffmpeg_cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-fflags', '+genpts',
            '-f', 'rawvideo', '-pix_fmt', input_pix_fmt, '-s', resolution, '-r', safe_float(fps),
            '-i', 'pipe:0', '-i', temp_audio_file, '-vf', v_filter,
            '-c:v', 'libx265', '-preset', 'veryslow', '-x265-params', ":".join(x265_cfg.values()),
            '-profile:v', 'main', '-pix_fmt', output_pix_fmt, '-color_range', 'pc', '-colorspace', 'bt709',
            '-color_primaries', 'bt709', '-color_trc', 'iec61966-2-1',
            '-c:a', 'libopus', '-b:a', AUDIO_BITRATE, '-ac', '1', '-vbr', 'on', '-fps_mode', 'vfr',
            '-movflags', '+faststart', output_file
        ]

        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        buffered_stdin = io.BufferedWriter(process.stdin, buffer_size=io_buffer_size)

        start_sec = float(START_TIME) if START_TIME else 0.0
        first_start = merged[0][0] if merged else start_sec
        starts = [s for s, e in merged]

        # Recalculate remaining frame counts to keep progression metrics accurate
        total_remaining_sec = max(0.0, total_sec - start_sec)
        v_total = int(total_remaining_sec * fps)

        # --- Frame Piping Loop (Run-Length De-Noised Content-Adaptive Selection) ---
        h_blocks, w_blocks = height // 8, width // 8
        total_blocks = h_blocks * w_blocks
        best_n_blocks = int(round(mpdecimate_frac * total_blocks))

        # Dynamic run tracking variables
        run_length = 0
        best_pixels = None
        best_score = float('inf')
        ref_frame_luma = None

        def get_cleanliness_score(luma_arr):
            # Compute horizontal and vertical gradients
            diff_h = np.abs(luma_arr[1:, :] - luma_arr[:-1, :])
            diff_w = np.abs(luma_arr[:, 1:] - luma_arr[:, :-1])

            # 1. Dynamic Noise Threshold (derived from the video's calibrated block noise floor)
            t_noise = max(1.0, best_lo / 64.0)

            # 2. Dynamic Edge Threshold (edges must reside significantly above the noise floor)
            t_edge = 6.0 * t_noise

            # 3. Dynamic Regularization Scale (derived from global contrast ratio & change entropy)
            alpha = 0.1 * Phi * (1.0 - H_norm)

            # Background Noise: Sub-pixel fluctuations, dither, mosquito noise
            noise_h_mask = (diff_h > 0) & (diff_h <= t_noise)
            noise_w_mask = (diff_w > 0) & (diff_w <= t_noise)
            noise_energy = (np.sum(diff_h[noise_h_mask]) + np.sum(diff_w[noise_w_mask])) / luma_arr.size

            # Edge Sharpness: High-contrast boundaries of vector text and UI elements
            edge_h_mask = (diff_h >= t_edge)
            edge_w_mask = (diff_w >= t_edge)
            edge_energy = (np.sum(diff_h[edge_h_mask]) + np.sum(diff_w[edge_w_mask])) / luma_arr.size

            # Combine: Minimize flat-area noise while Maximizing sharp edges. Lower score is cleaner!
            return noise_energy - (alpha * edge_energy)

        with tqdm(total=v_total, unit="frame", desc="Encoding") as pbar:
            container.seek(int(first_start / float(v_stream.time_base)), stream=v_stream)

            for frame in container.decode(v_stream):
                ts = float(frame.pts * v_stream.time_base) if frame.pts is not None else 0.0
                if ts < first_start:
                    continue  # Safely fast-forward past keyframe boundaries to target seek start

                idx = bisect.bisect_right(starts, ts)
                if idx > 0 and ts <= merged[idx - 1][1]:
                    # 1. Decode the frame pixels to the target format exactly once
                    frame_pixels = frame.to_ndarray(format=pix_fmt)

                    # 2. Extract luma directly from the decoded ndarray (no extra C-buffer parsing)
                    arr_luma = extract_luma_from_ndarray(frame_pixels, height, width)

                    # Emulate mpdecimate logic to verify if a state change occurred
                    is_different = True
                    if ref_frame_luma is not None:
                        diff = np.abs(arr_luma - ref_frame_luma)
                        if np.max(diff) > 0:
                            blocks = diff[:h_blocks * 8, :w_blocks * 8].reshape(h_blocks, 8, w_blocks, 8)
                            block_sums = blocks.sum(axis=(1, 3)).flatten()

                            num_hi = np.sum(block_sums > best_hi)
                            num_lo = np.sum(block_sums > best_lo)
                            is_different = (num_hi > 0) or (num_lo > best_n_blocks)
                        else:
                            is_different = False

                    # Score the current frame's visual cleanliness
                    score = get_cleanliness_score(arr_luma)

                    if is_different:
                        # Flush the accumulated de-noised run to FFmpeg standard input
                        if run_length > 0 and best_pixels is not None:
                            for _ in range(run_length):
                                buffered_stdin.write(memoryview(best_pixels))

                        # Initialize a new static run (copy lazily only when selected)
                        run_length = 1
                        best_pixels = np.ascontiguousarray(frame_pixels)
                        best_score = score
                        ref_frame_luma = arr_luma
                    else:
                        # Duplicate detected; increment run length and check if this frame is cleaner
                        run_length += 1
                        if score < best_score:
                            best_pixels = np.ascontiguousarray(frame_pixels)
                            best_score = score

                pbar.update(1)
                if pbar.format_dict['rate']:
                    pbar.set_postfix_str("Speed: " + str(safe_float(pbar.format_dict['rate'] / fps)) + "x")

            # Flush the final trailing run after the decode loop completes
            if run_length > 0 and best_pixels is not None:
                for _ in range(run_length):
                    buffered_stdin.write(memoryview(best_pixels))

        buffered_stdin.close()
        process.wait()
        container.close()

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_file:
            log_file.close()
            print(f"Log written to: {log_file_path}")


if __name__ == "__main__":
    run_pipeline()