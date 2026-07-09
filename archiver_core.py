"""
archiver_core.py

A physical, data-driven screen recording archival pipeline.
"""

import av
import numpy as np
import subprocess
import sys
import os
from tqdm import tqdm
import io
import json
import bisect
import re
import time
import ast
from fractions import Fraction
from decimal import Decimal
from concurrent.futures import ProcessPoolExecutor

# ==============================================================================
# --- CONFIGURATION & PHYSICAL CONSTANTS ---
# ==============================================================================

# Time seeking parameters
START_TIME = 0.0            # Default start time. Can be overridden via command-line or restored from log.

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
MIN_PHYSICAL_LO = BLOCK_SIZE
MAX_PHYSICAL_HI = BLOCK_SIZE * MAX_PIXEL_DIFF

# 255 discrete thresholds corresponding to average pixel differences of 1.0 to 255.0 steps
THRESHOLDS = np.arange(1, 256) * BLOCK_SIZE
EPSILON = np.finfo(np.float64).eps  # Machine epsilon used to prevent division-by-zero errors
BFRAMES_LIMIT = 16                  # Physical upper limit of consecutive B-frames
MAX_RC_LOOKAHEAD = 250              # Upper limit for x265 rate control lookahead buffer

# --- Analytical Algorithmic Complexity Crossover ---
CROSSOVER_LIMIT = HIST_SIZE // (THRESHOLDS.size - 1)  # Exactly 64 blocks


def safe_float(val):
    """
    Converts float values to their shortest, lossless decimal representation
    bypassing system locales. Guarantees 17-digit round-trip safety 
    without exposing binary approximation noise.
    """
    return str(float(val))


def high_precision_float(val):
    """
    Maintained for API naming consistency. Performs a lossless 
    double-precision round-trip conversion.
    """
    return str(float(val))


def to_mixed_fraction_string(num):
    """
    Utility helper to convert floats into mixed fractions.
    """
    whole = int(num)
    f_part = Fraction(num - whole).limit_denominator(10000)
    if f_part == 0:
        return f"{whole}"
    elif f_part == 1:
        return f"{whole + 1}"
    else:
        return f"{whole} {f_part.numerator}/{f_part.denominator}"


def parse_mixed_fraction(val):
    """
    Parses a mixed fraction string (e.g., "11219 83/150" or "89/750") 
    or a standard numeric value back into a double-precision float.
    """
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    if ' ' in val_str:
        whole, frac = val_str.split(' ', 1)
        num, den = frac.split('/')
        return float(whole) + float(num) / float(den)
    elif '/' in val_str:
        num, den = val_str.split('/')
        return float(num) / float(den)
    else:
        return float(val_str)


def format_hhmmss_decimal(sec_val):
    """
    Converts seconds to HH:MM:SS.SS... format without truncating non-zero decimals,
    using decimal.Decimal for precision to avoid binary floating-point noise.
    """
    d = Decimal(str(sec_val))
    hours = int(d // 3600)
    minutes = int((d % 3600) // 60)
    seconds = d % 60
    
    sec_str = str(seconds.normalize())
    if '.' in sec_str:
        parts = sec_str.split('.')
        sec_formatted = f"{parts[0].zfill(2)}.{parts[1]}"
    else:
        sec_formatted = sec_str.zfill(2)
        
    return f"{hours:02d}:{minutes:02d}:{sec_formatted}"


def seconds_to_time_string(sec_val):
    """
    Converts a raw float duration into a clean, readable time representation
    (e.g., 1032.0 -> '17:12', 3661.0 -> '01:01:01').
    """
    sec_val = float(sec_val)
    hours = int(sec_val // 3600)
    minutes = int((sec_val % 3600) // 60)
    seconds = sec_val % 60
    
    sec_str = format(seconds, '.15g')
    if '.' in sec_str:
        parts = sec_str.split('.')
        sec_formatted = f"{parts[0].zfill(2)}.{parts[1]}"
    else:
        sec_formatted = sec_str.zfill(2)
        
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec_formatted}"
    else:
        return f"{minutes:02d}:{sec_formatted}"


def parse_time_string_to_seconds(time_str):
    """
    Parses an interactive input time string (e.g., '21:25', '01:10:05', or raw seconds)
    back into a double-precision float.
    """
    time_str = str(time_str).strip()
    if not time_str:
        return 0.0
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return float(m) * 60 + float(s)
        else:
            raise ValueError(f"Invalid time format: {time_str}")
    else:
        return float(time_str)


def extract_luma_from_ndarray(arr, height, width):
    """
    Extracts the Y (Luma) channel as a 2D ndarray of shape (height, width)
    directly from the decoded frame array to avoid redundant memory copies.
    """
    if arr.ndim == 2:
        return arr[:height, :].astype(np.int16)
    elif arr.ndim == 3:
        if arr.shape[0] == 3:
            return arr[0, :, :].astype(np.int16)
        elif arr.shape[2] == 3:
            return (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.int16)
        elif arr.shape[2] == 1:
            return arr[:, :, 0].astype(np.int16)
    return arr[:height, :width].astype(np.int16)


# ==============================================================================
# --- INTERACTIVE TERMINAL UTILITIES --------
# ==============================================================================

def timed_input(prompt, timeout):
    """
    Cross-platform non-blocking timed input.
    """
    print(prompt, end='', flush=True)
    
    if sys.platform == 'win32':
        import msvcrt
        start_time = time.time()
        input_chars = []
        while time.time() - start_time < timeout:
            if msvcrt.kbhit():
                char = msvcrt.getwche()
                if char in ('\r', '\n'):
                    print()
                    return ''.join(input_chars)
                elif char == '\b':
                    if input_chars:
                        input_chars.pop()
                        print(' \b', end='', flush=True)
                else:
                    input_chars.append(char)
            time.sleep(0.05)
        print("\n\n[Timeout] No input received. Proceeding automatically...")
        return None
    else:
        import select
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.readline().strip()
        else:
            print("\n\n[Timeout] No input received. Proceeding automatically...")
            return None


def write_state_to_log(output_file, payload):
    """
    Appends a machine-readable structured JSON state line directly into the log file.
    No console outputs ever touch this file, keeping it 100% clean of terminal logs.
    """
    log_file_path = output_file + ".log"
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"#STATE: {json.dumps(payload)}\n")
    except Exception:
        pass


def parse_existing_log(log_file_path):
    """
    Parses an existing log file to extract high-precision history and check convergence.
    Safely converts keep intervals (even if saved as mixed fractions) back into floats.
    """
    if not os.path.exists(log_file_path):
        return None
        
    try:
        history_lo = []
        history_n = []
        projections = []
        state_matches = []
        merged_list = None
        total_sec_val = None
        start_time_val = 0.0
        
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith("#STATE:"):
                    json_str = line.strip().split("#STATE:", 1)[1].strip()
                    try:
                        data = json.loads(json_str)
                        if data['event'] == 'intervals':
                            raw_merged = data['merged']
                            merged_list = []
                            for start_val, end_val in raw_merged:
                                merged_list.append((parse_mixed_fraction(start_val), parse_mixed_fraction(end_val)))
                            total_sec_val = data['total_sec']
                            start_time_val = data.get('start_time', 0.0)
                        elif data['event'] == 'iteration':
                            history_lo.append(data['lo'])
                            history_n.append(data['n'])
                            state_matches.append(data['state'])
                        elif data['event'] == 'projection':
                            projections.append((data['lo_est'], data['n_est']))
                    except Exception:
                        pass
                        
        if history_lo:
            converged = False
            final_lo, final_n = None, None
            final_state = None
            
            if len(projections) >= 2:
                last_lo = projections[-1][0]
                last_n = projections[-1][1]
                second_last_lo = projections[-2][0]
                second_last_n = projections[-2][1]
                
                last_lo_rounded = int(round(last_lo))
                last_n_rounded = int(round(last_n))
                second_last_lo_rounded = int(round(second_last_lo))
                second_last_n_rounded = int(round(second_last_n))
                
                if last_lo_rounded == second_last_lo_rounded and last_n_rounded == second_last_n_rounded:
                    converged = True
                    final_lo = last_lo_rounded
                    final_n = last_n_rounded
                    if state_matches:
                        final_state = state_matches[-1]
                        
            return {
                'history_lo': history_lo,
                'history_n': history_n,
                'converged': converged,
                'final_lo': final_lo,
                'final_n': final_n,
                'final_state': final_state,
                'merged': merged_list,
                'total_sec': total_sec_val,
                'start_time': start_time_val
            }
    except Exception as e:
        print(f"[RESUME] Warning: Failed to parse existing log file ({e}). Starting fresh.")
        
    return None


# ==============================================================================
# --- MATHEMATICAL CLASS SEPARATION --------
# ==============================================================================

def perform_otsu_sweep(data, weights=None):
    """
    Class 3 Otsu Thresholding Sweep.
    """
    if data.size == 0:
        return 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    d_min, d_max = np.min(data), np.max(data)
    if np.isclose(d_min, d_max, atol=EPSILON):
        return float(d_min), float(d_max), 0.0, (float(d_min), float(d_min), float(d_min)), (1.0, 0.0, 0.0)

    hist, bin_edges = np.histogram(data, bins=256, range=(d_min, d_max), weights=weights)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total_samples = np.sum(hist)
    if total_samples < EPSILON:
        return float(d_min), float(d_max), 0.0, (float(d_min), float(d_min), float(d_min)), (1.0, 0.0, 0.0)

    prob = hist.astype(np.float64) / total_samples

    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_centers)
    mu_total = mu[-1]

    total_var = np.sum(prob * (bin_centers - mu_total) ** 2)

    max_var_b = -1.0
    best_t1, best_t2 = 0, 0

    for t1 in range(len(prob) - 2):
        w0 = omega[t1]
        if w0 < EPSILON: continue
        m0 = mu[t1] / w0

        for t2 in range(t1 + 1, len(prob) - 1):
            w1 = omega[t2] - w0
            w2 = 1.0 - omega[t2]

            if w1 < EPSILON or w2 < EPSILON: continue

            m1 = (mu[t2] - mu[t1]) / w1
            m2 = (mu_total - mu[t2]) / w2

            var_b = (w0 * (m0 ** 2) + w1 * (m1 ** 2) + w2 * (m2 ** 2)) - (mu_total ** 2)

            if var_b > max_var_b:
                max_var_b, best_t1, best_t2 = var_b, t1, t2

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
    Decodes and profiles the audio stream to extract speech metrics.
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

            if np.issubdtype(raw.dtype, np.integer):
                samples /= np.float64(np.iinfo(raw.dtype).max)

            if samples.size < 2:
                pbar.update(frame.samples / sr)
                continue

            rms = np.sqrt(np.mean(samples ** 2))
            zcr = np.mean(np.abs(np.diff(np.sign(samples)))) / 2

            fft_data = np.abs(np.fft.rfft(samples))
            freqs = np.fft.rfftfreq(len(samples), 1 / sr)
            total_e = np.sum(fft_data)

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

    non_zero_rms = rms_vals[rms_vals > 0]
    if len(non_zero_rms) == 0:
        auto_rms_thresh = 0.0
    else:
        min_audio_magnitude = np.min(non_zero_rms)
        safe_rms = np.clip(rms_vals, a_min=min_audio_magnitude, a_max=None)
        t1_rms, t2_rms, _, _, _ = perform_otsu_sweep(np.log10(safe_rms))
        auto_rms_thresh = 10 ** t2_rms

    t1_zcr, t2_zcr, _, _, _ = perform_otsu_sweep(zcr_vals)
    auto_zcr_thresh = t1_zcr

    t1_fft, t2_fft, _, _, _ = perform_otsu_sweep(fft_vals)
    auto_fft_thresh = t2_fft

    print(f"  > Audio Thresholds: RMS={auto_rms_thresh}, ZCR={auto_zcr_thresh}, FFT={auto_fft_thresh}")

    start_sec = float(START_TIME) if START_TIME else 0.0
    frame_dur = 1.0 / fps
    keep_intervals, current_start = [], None

    for f in audio_stats:
        is_speech = (f['rms'] > auto_rms_thresh) and (f['zcr'] < auto_zcr_thresh) and (f['fft'] > auto_fft_thresh)
        if is_speech:
            if current_start is None: current_start = f['ts']
        elif current_start is not None:
            keep_intervals.append((max(start_sec, current_start - MARGIN_SECS),
                                   min(dur, f['ts'] + MARGIN_SECS)))
            current_start = None
    if current_start is not None:
        keep_intervals.append((max(start_sec, current_start - MARGIN_SECS), dur))

    if not keep_intervals:
        return [(start_sec, start_sec + frame_dur)]

    keep_intervals.sort()
    merged = [keep_intervals[0]]
    for next_s, next_e in keep_intervals[1:]:
        if next_s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], next_e))
        else:
            merged.append((next_s, next_e))

    print(f"  > Keep Intervals: {merged}")
    return merged


# ==============================================================================
# --- PHASE 3: VIDEO PROFILING ENGINE ------
# ==============================================================================

def worker_analyze_sweep(args):
    """
    Parallel Worker Process.
    """
    input_file, segments, filter_hi, filter_lo, filter_n = args
    frame_stats = []
    activity_log = []
    local_active_blocks = 0
    local_total_blocks = 0
    local_frames_processed = 0

    local_mag_hist = np.zeros(HIST_SIZE, dtype=np.int64)

    try:
        container = av.open(input_file)
        v_stream = container.streams.video[0]

        v_stream.codec_context.thread_count = 0
        v_stream.codec_context.thread_type = "AUTO"

        for start_time, end_time in segments:
            container.seek(int(start_time / float(v_stream.time_base)), stream=v_stream)
            ref_frame = None

            for frame in container.decode(v_stream):
                ts = float(frame.pts * v_stream.time_base) if frame.pts is not None else 0.0
                if ts < start_time:
                    continue
                if ts > end_time: break
                local_frames_processed += 1

                plane = frame.planes[0]
                if len(frame.planes) > 1:
                    arr = np.frombuffer(plane, np.uint8).reshape(plane.height, plane.line_size)
                    curr_frame = arr[:, :plane.width].astype(np.int16)
                else:
                    curr_frame = extract_luma_from_ndarray(frame.to_ndarray(), plane.height, plane.width)

                if ref_frame is None:
                    ref_frame = curr_frame
                    continue

                diff = np.abs(curr_frame - ref_frame)
                if np.max(diff) == 0:
                    continue

                h, w = curr_frame.shape
                num_blocks = (h // 8) * (w // 8)
                local_total_blocks += num_blocks

                blocks = diff[:(h // 8) * 8, :(w // 8) * 8].reshape(h // 8, 8, w // 8, 8)
                block_sums = blocks.sum(axis=(1, 3)).flatten()
                active_sums = block_sums[block_sums > 0]

                if active_sums.size > 0:
                    local_active_blocks += active_sums.size
                    activity_log.append((ts, np.max(active_sums)))

                    if active_sums.size < CROSSOVER_LIMIT:
                        counts = np.sum(active_sums[:, None] > THRESHOLDS, axis=0)
                    else:
                        bincount = np.bincount(active_sums, minlength=HIST_SIZE)
                        suffix_sum = np.cumsum(bincount[::-1])[::-1]
                        suffix_sum_padded = np.append(suffix_sum, 0)
                        counts = suffix_sum_padded[THRESHOLDS + 1]

                    frame_stats.append(counts.astype(np.int32))
                    np.add.at(local_mag_hist, active_sums, 1)

                    num_hi = np.sum(block_sums > filter_hi)
                    num_lo = np.sum(block_sums > filter_lo)
                    is_kept = (num_hi > 0) or (num_lo > filter_n)

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
    Calculates the normalized Shannon Entropy of active changes.
    """
    if counts_data.shape[0] == 0 or g_total == 0:
        return float(MAX_PHYSICAL_HI), 100.0, 1.0, 0.0, 0.0, 0.0, 0.0

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

    # HI is strictly locked to the absolute physical maximum of an 8x8 block (16320.0)
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
        n_blocks = float(perform_otsu_sweep(active_frame_counts)[0])

    return best_hi, best_lo, n_blocks, separability, omega_metric, phi_metric, norm_entropy


def profile_video_sweep(input_file, keep_intervals, width, height, output_file, parsed_log=None):
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
            for packet in kf_container.demux(v_stream_kf):
                if packet.pts is None:
                    continue
                if packet.is_keyframe:
                    kf_times.append(float(packet.pts * v_stream_kf.time_base))
    except Exception as e:
        print(f"  > Warning: Rapid keyframe pre-scan failed ({e}). Falling back to empty GOP map.")
        kf_times = []

    kf_times.sort()
    total_blocks = (width // 8) * (height // 8)

    # ==========================================================================
    # --- RESUME POINT CHECK: DIRECT LOAD ---
    # ==========================================================================
    if parsed_log and parsed_log['converged']:
        best_lo = parsed_log['final_lo']
        best_hi = float(MAX_PHYSICAL_HI)  # Strictly locked to physical maximum
        best_n = parsed_log['final_n']

        if parsed_log['final_state']:
            separability, norm_entropy, omega_metric, phi_metric = parsed_log['final_state']
        else:
            separability, norm_entropy, omega_metric, phi_metric = 0.85, 0.68, 0.003, 0.97

        mpdecimate_frac = np.clip(best_n / total_blocks, EPSILON, 1.0)

        print(f"\n[RESUME] Native Convergence Found in Log. Direct Loading Parameters:")
        print(f"  > Parameters: LO={best_lo}, HI={best_hi}, n={best_n} (FRAC={safe_float(mpdecimate_frac)})")
        return best_hi, best_lo, mpdecimate_frac, separability, omega_metric, phi_metric, norm_entropy, []

    def extrapolate_infinite_limit(seq, min_val=0, max_val=MAX_BLOCK_SUM):
        if len(seq) < 5:
            return None, None, None, None

        y = np.array(seq, dtype=np.float64)
        X = np.column_stack([y[1:-1], y[:-2], np.ones(len(y) - 2)])
        Y = y[2:]

        coeffs, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
        if rank < 3:
            return None, None, None, None

        A1, A2, c = coeffs

        # Schur-Cohn Stability Check
        is_stable = (A2 > -1.0) and (A1 + A2 < 1.0) and (A2 - A1 < 1.0)
        if not is_stable:
            return None, None, A1, A2

        denom = 1.0 - A1 - A2
        if abs(denom) < 1e-8:
            return None, None, A1, A2

        limit = c / denom
        if not (min_val <= limit <= max_val):
            return None, None, A1, A2

        roots = np.roots([1, -A1, -A2])
        spectral_radius = np.max(np.abs(roots)) if 'roots' in locals() else np.sqrt(max(0.0, -A2))
        if distance > 0.5 and spectral_radius > EPSILON:
            steps = int(np.ceil(np.log(0.5 / distance) / np.log(spectral_radius)))
        else:
            steps = 0

        return limit, steps, A1, A2

    logical_cores = os.cpu_count() or 1
    num_workers = max(1, int(np.round(np.sqrt(logical_cores))))

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

    T_min = total_keep_dur / (16.0 * num_workers)
    T_max = total_keep_dur / (2.0 * num_workers)

    if active_gop_durs.size >= 10:
        log_gops = np.log10(np.clip(active_gop_durs, a_min=0.1, a_max=None))
        _, t2_log, _, _, _ = perform_otsu_sweep(log_gops)
        T_static = 10 ** t2_log
    else:
        T_static = np.sqrt(T_min * T_max)

    T_static = np.clip(T_static, T_min, T_max)

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
        f"  > Divided keep intervals ({high_precision_float(total_keep_dur)}s active) into {len(groups)} hardware-scaled segments "
        f"({num_workers} workers, avg chunk: {high_precision_float(avg_chunk_dur)}s)...")

    # ==========================================================================
    # --- RESUME POINT CHECK: HISTORY INJECTION ---
    # ==========================================================================
    lo_history = []
    n_history = []
    start_iter_idx = 0
    lo, n_blocks = 0.0, 0.0

    if parsed_log and not parsed_log['converged']:
        lo_history = list(parsed_log['history_lo'])
        n_history = list(parsed_log['history_n'])
        start_iter_idx = len(lo_history)
        lo = lo_history[-1]
        n_blocks = n_history[-1]
        print(f"  > Pre-loaded {start_iter_idx} steps of historical AR(2) metrics from log.")

    prev_extrapolated_lo = None
    prev_extrapolated_n = None
    hi = float(MAX_PHYSICAL_HI)  # Locked to physical maximum for workers
    max_iters_current = 24
    relaxation_damping = 2.0 / 3.0
    static_chunks = set()
    iter_durations = []

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

    # Initialized before loop to act as persistent state flag
    converged_early = False
    prev_energy_lo = None

    iter_idx = start_iter_idx
    while iter_idx < max_iters_current:
        iter_start = time.time()
        active_tasks = []
        task_mapping = []

        for i, t in enumerate(tasks):
            if i in static_chunks: continue
            active_tasks.append((t[0], t[1], hi, lo, n_blocks))
            task_mapping.append(i)

        if not active_tasks: break

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

        hi_new, lo_new, n_new, separability, omega_metric, phi_metric, norm_entropy_new = calculate_radiometric_constants_raw(
            counts_data, global_mag_hist, g_act, g_tot
        )

        last_iter_metrics = {
            'separability': separability,
            'omega_metric': omega_metric,
            'phi_metric': phi_metric,
            'norm_entropy': norm_entropy_new,
            'activity_log': global_activity_log,
            'g_act': g_act,
            'g_tot': g_tot,
            'g_frm': g_frm
        }

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

        write_state_to_log(output_file, {
            "event": "iteration",
            "iter": iter_idx + 1,
            "lo": float(lo),
            "n": float(n_blocks),
            "state": [float(separability), float(norm_entropy_new), float(omega_metric), float(phi_metric)]
        })

        delta_str_lo = f" (Δ abs: {high_precision_float(lo_diff_abs)}, rel: {high_precision_float(lo_diff_rel)}%)" if iter_idx > 0 else " (Initial)"
        delta_str_n = f" (Δ abs: {high_precision_float(n_diff_abs)}, rel: {high_precision_float(n_diff_rel)}%)" if iter_idx > 0 else " (Initial)"

        print(
            f"    - Iteration {iter_idx + 1}:\n"
            f"      > Parameters: LO={high_precision_float(lo)}{delta_str_lo}\n"
            f"                    HI={high_precision_float(hi)} (Locked)\n"
            f"                    n={high_precision_float(n_blocks)}{delta_str_n}\n"
            f"      > State:      Sep={high_precision_float(separability)}, H_norm={high_precision_float(norm_entropy_new)}, "
            f"Omega={high_precision_float(omega_metric)}, Phi={high_precision_float(phi_metric)}"
        )

        if len(lo_history) >= 5:
            lo_est, lo_steps, lo_A1, lo_A2 = extrapolate_infinite_limit(lo_history, min_val=MIN_PHYSICAL_LO)
            n_est, n_steps, n_A1, n_A2 = extrapolate_infinite_limit(n_history, max_val=total_blocks)

            # Stage 1: Active Divergence Abort Check
            if (len(lo_history) >= 6) and (lo_est is None or n_est is None):
                print(f"    - [EARLY ABORT] System instability detected (Divergent AR(2) coefficients). Halting sweep.")
                break

            if lo_est is not None and n_est is not None:
                # Stage 2: Lyapunov Energy Monotonicity Check
                e_curr_lo = lo_history[-1] - lo_est
                e_prev_lo = lo_history[-2] - lo_est
                energy_lo = (e_curr_lo ** 2) - (lo_A2 * (e_prev_lo ** 2))

                if prev_energy_lo is not None and energy_lo > prev_energy_lo:
                    if energy_lo / prev_energy_lo > 1.05:  # Abort if energy gain exceeds a 5% noise tolerance threshold
                        print(
                            f"    - [EARLY ABORT] Lyapunov Energy violation (E_curr: {energy_lo:.4f} > E_prev: {prev_energy_lo:.4f}). System gaining entropy. Halting sweep.")
                        break

                prev_energy_lo = energy_lo

                print(f"    > Current Projection L(n): LO={high_precision_float(lo_est)} (est. {lo_steps} iters) | "
                      f"n={high_precision_float(n_est)} (est. {n_steps} iters)")

                write_state_to_log(output_file, {
                    "event": "projection",
                    "lo_est": float(lo_est),
                    "n_est": float(n_est)
                })

                lo_est_rounded = int(round(lo_est))
                n_est_rounded = int(round(n_est))

                if prev_extrapolated_lo is not None and prev_extrapolated_n is not None:
                    prev_lo_rounded = int(round(prev_extrapolated_lo))
                    prev_n_rounded = int(round(prev_extrapolated_n))

                    if lo_est_rounded == prev_lo_rounded and n_est_rounded == prev_n_rounded:
                        print(
                            f"    - Extrapolated limit converged early at Iteration {iter_idx + 1} (LO={lo_est_rounded}, n={n_est_rounded}). Halting loop.")
                        converged_early = True
                        break

                prev_extrapolated_lo = lo_est
                prev_extrapolated_n = n_est
            else:
                print("    > Current Projection L(n): Gathering stable history (Extrapolation Unstable/Unavailable)")
                prev_extrapolated_lo = None
                prev_extrapolated_n = None

        iter_durations.append(time.time() - iter_start)
        iter_idx += 1

        if iter_idx >= max_iters_current and not converged_early:
            avg_dur = np.mean(iter_durations) if iter_durations else 15.0

            print("\a", end='', flush=True)
            print("\n" + "!" * 80)
            print("!!! ATTENTION: CALIBRATION ITERATIONS REACHED LIMIT WITHOUT CONVERGENCE !!!")
            print(
                f"  > Terminal will pause for up to {format_hhmmss_decimal(avg_dur)} (average step duration) for response.")
            print("! Press Enter (with no input) to skip and finalize using the current state.")
            print("!" * 80 + "\n")

            prompt = f"Enter additional iterations to run (or press Enter to skip): "
            user_input = timed_input(prompt, timeout=avg_dur)

            if user_input is not None and user_input.strip().isdigit():
                extend_by = int(user_input.strip())
                if extend_by > 0:
                    max_iters_current += extend_by
                    print(f"\n[RESUME] Extending limit by +{extend_by} steps (New limit: {max_iters_current})...\n")
                else:
                    print("\n[RESUME] Invalid amount. Finalizing...")
            else:
                print("\n[RESUME] Finalizing calculations using current parameters...")

    # Final calculations using our pure AR(2) model
    lo_extrapolated, _, _, _ = extrapolate_infinite_limit(lo_history, min_val=MIN_PHYSICAL_LO)
    n_blocks_extrapolated, _, _, _ = extrapolate_infinite_limit(n_history, max_val=total_blocks)

    if converged_early:
        print(f"\n  > Global Sequence Extrapolation Converged Successfully.")
        best_lo = int(round(lo_extrapolated))
        best_n = int(round(n_blocks_extrapolated))
    else:
        if lo_extrapolated is not None and n_blocks_extrapolated is not None:
            print(
                f"\n  > Calibration loop completed without full convergence (Using stable mathematical extrapolation limits).")
            best_lo = int(round(lo_extrapolated))
            best_n = int(round(n_blocks_extrapolated))
        else:
            print(
                f"\n  > Calibration loop bypassed (Extrapolation Unstable/Unavailable). Finalizing with empirical parameters.")
            best_lo = int(round(lo_history[-1]))
            best_n = int(round(n_history[-1]))

    # Strictly locked to physical maximum
    best_hi = float(MAX_PHYSICAL_HI)
    print(f"  > Infinite-Limit Fixed Point: LO={best_lo}, HI={best_hi} (Locked), n={best_n}")

    separability = last_iter_metrics['separability']
    omega_metric = last_iter_metrics['omega_metric']
    phi_metric = last_iter_metrics['phi_metric']
    norm_entropy = last_iter_metrics['norm_entropy']
    activity_log = last_iter_metrics['activity_log']
    g_tot = last_iter_metrics['g_tot']
    g_frm = last_iter_metrics['g_frm']

    total_blocks_per_frame = (g_tot / g_frm) if g_frm > 0 else 1
    mpdecimate_frac = np.clip(best_n / total_blocks_per_frame, EPSILON, 1.0)

    print(f"\n--- Radiometric Optimization Results ---")
    print(
        f"  > Optimal HI={best_hi}, LO={best_lo}, n={best_n} (FRAC={high_precision_float(mpdecimate_frac)}), Sep={high_precision_float(separability)}")

    return best_hi, best_lo, mpdecimate_frac, separability, omega_metric, phi_metric, norm_entropy, activity_log


# ==============================================================================
# --- TEMPORAL & AUDIO EXPORT HELPERS ------
# ==============================================================================

def calculate_temporal_lookahead(merged_intervals, fps):
    """
    Calculates the optimal x265 rate control lookahead buffer based on keep intervals.
    """
    durations = np.array([(e - s) * fps for s, e in merged_intervals])
    if len(durations) < 3:
        return MAX_RC_LOOKAHEAD

    t1_t, _, t_sep, _, _ = perform_otsu_sweep(durations)
    signal_mask = durations > t1_t
    if t_sep < EPSILON or not np.any(signal_mask):
        return MAX_RC_LOOKAHEAD

    auto_lookahead_raw = np.mean(durations[signal_mask])
    return int(np.clip(round(auto_lookahead_raw), BFRAMES_LIMIT, MAX_RC_LOOKAHEAD))


def export_trimmed_audio(input_file, output_wav_path, merged_intervals, total_sec):
    """
    Extracts, resamples, and multiplexes the audio segments into a unified WAV file.
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
            if ts < start_sec: continue

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
    """
    global START_TIME
    if len(sys.argv) < 4:
        print("Error: Missing execution arguments.")
        print("Usage: python archiver_core.py <input_file> <output_file> <temp_audio_file> [start_time]")
        sys.exit(1)

    input_file, output_file, temp_audio_file = sys.argv[1], sys.argv[2], sys.argv[3]

    # Pre-load log file to check for stored START_TIME and intervals
    log_file_path = output_file + ".log"
    parsed_log = parse_existing_log(log_file_path)

    stored_start_time = 0.0
    if parsed_log and 'start_time' in parsed_log:
        stored_start_time = parsed_log['start_time']

    # Resolve START_TIME (CLI argument takes precedence, otherwise prompts interactively)
    if len(sys.argv) >= 5:
        try:
            START_TIME = parse_time_string_to_seconds(sys.argv[4])
            print(f"  > CLI Override: START_TIME set to {seconds_to_time_string(START_TIME)} ({high_precision_float(START_TIME)} seconds).")
        except ValueError:
            print(f"Warning: Invalid start_time argument '{sys.argv[4]}'. Defaulting to 00:00.")
            START_TIME = 0.0
    else:
        default_str = seconds_to_time_string(stored_start_time)
        prompt_str = f"Enter start time (e.g., 21:25, 01:10:05, or raw seconds. Press Enter for {default_str}): "
        print(prompt_str, end='', flush=True)
        user_input = sys.stdin.readline().strip()
        
        if user_input:
            try:
                START_TIME = parse_time_string_to_seconds(user_input)
                print(f"  > START_TIME set to {seconds_to_time_string(START_TIME)} ({high_precision_float(START_TIME)} seconds).")
            except ValueError:
                print(f"Warning: Invalid time format entered. Defaulting to {default_str}.")
                START_TIME = stored_start_time
        else:
            START_TIME = stored_start_time
            print(f"  > Using default START_TIME: {default_str} ({high_precision_float(START_TIME)} seconds).")

    # Open log descriptor safely on resume. No Tee redirection means the log 
    # file stays 100% clean, containing only structured #STATE JSON entries.
    log_mode = 'w'
    if parsed_log:
        log_mode = 'a'
        if parsed_log['converged'] and parsed_log['merged'] is not None and parsed_log['total_sec'] is not None:
            print(f"[RESUME] Existing log converged at LO={parsed_log['final_lo']}, n={parsed_log['final_n']}. Bypassing calibration sweep.")
        else:
            print(f"[RESUME] Existing log has {len(parsed_log['history_lo'])} unconverged steps. Resuming sweep in append mode.")

    has_audio = False

    try:
        # Pre-scan Stream Metadata
        container = av.open(input_file)
        v_stream = container.streams.video[0]
        fps = float(v_stream.average_rate) if v_stream.average_rate else 10.0
        if fps <= 0: fps = 10.0
        width = v_stream.width
        height = v_stream.height
        pix_fmt = v_stream.pix_fmt
        has_audio = len(container.streams.audio) > 0
        container.close()

        # Check and Load Cached Intervals
        use_cached_intervals = False
        if parsed_log and parsed_log['merged'] is not None and parsed_log['total_sec'] is not None:
            use_cached_intervals = True

        if use_cached_intervals:
            merged = parsed_log['merged']
            total_sec = parsed_log['total_sec']
            print(f"[RESUME] Loaded cached intervals from log ({len(merged)} segments, {high_precision_float(total_sec)} total seconds). "
                  f"Skipping Phase 1 (Audio Stats) and Phase 2 (Interval Generation).")
        else:
            if has_audio:
                # --- Executing Phase 1 (Audio Stats) ---
                audio_stats, total_sec = collect_audio_stats(input_file)

                # --- Executing Phase 2 (Speech-Driven Keep Intervals) ---
                merged = determine_intervals(audio_stats, total_sec, fps)
            else:
                container = av.open(input_file)
                total_sec = float(container.duration / av.time_base) if container.duration else 0.0
                container.close()
                merged = [(START_TIME if START_TIME else 0.0, total_sec)]
                print(f"  > Keep Intervals: {merged}")

            # Serialize keep intervals and start time metadata
            write_state_to_log(output_file, {
                "event": "intervals",
                "merged": [[to_mixed_fraction_string(s), to_mixed_fraction_string(e)] for s, e in merged],
                "total_sec": total_sec,
                "start_time": float(START_TIME) if START_TIME else 0.0
            })

            print(f"  > Total Duration: {high_precision_float(total_sec)} seconds")

        # --- Tier 2: Unified Video Profiling & Radiometric Calibration ---
        use_cached_calibration = False
        if parsed_log and parsed_log['converged']:
            use_cached_calibration = True

        if use_cached_calibration:
            best_lo = parsed_log['final_lo']
            best_n = parsed_log['final_n']
            best_hi = float(MAX_PHYSICAL_HI)
            
            if parsed_log['final_state']:
                S_val, H_norm, Omega, Phi = parsed_log['final_state']
            else:
                S_val, H_norm, Omega, Phi = 0.85, 0.68, 0.003, 0.97
                
            total_blocks = (width // 8) * (height // 8)
            mpdecimate_frac = np.clip(best_n / total_blocks, EPSILON, 1.0)
            
            print(f"[RESUME] Loaded Converged State: LO={best_lo}, HI={best_hi}, n={best_n} (FRAC={high_precision_float(mpdecimate_frac)})")
            activity_log = []
        else:
            best_hi, best_lo, mpdecimate_frac, S_val, Omega, Phi, H_norm, activity_log = profile_video_sweep(
                input_file, merged, width, height, output_file, parsed_log=parsed_log)

        # --- Phase 4: Master Encode Preparation ---
        container = av.open(input_file)
        v_stream = container.streams.video[0]

        v_stream.codec_context.thread_count = 0
        v_stream.codec_context.thread_type = "AUTO"

        auto_lookahead = calculate_temporal_lookahead(merged, fps)
        print(f"  > Temporal Result: Lookahead={auto_lookahead}")

        if has_audio:
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

        auto_ref = int(np.clip(np.round(S_val / (Omega + EPSILON)), 1, 6))
        b_stability_odds = (1.0 - Omega) / (Omega + EPSILON)
        b_bias_factor = np.tanh(np.log(b_stability_odds))
        auto_b_bias = int(np.clip(np.round(100.0 * b_bias_factor), -90, 100))
        auto_bframes = int(np.clip(np.round(BFRAMES_LIMIT * (1.0 - (Omega * H_norm))), 4, BFRAMES_LIMIT))
        auto_scenecut = int(np.clip(np.round(100.0 * S_val), 10, 90))
        psi = Phi * (1.0 - H_norm)
        auto_psy_rd = round(float(5.0 * psi), 2)
        auto_psy_rdoq = round(float(auto_psy_rd * 0.5), 2)
        auto_qcomp = round(float(np.clip(1.0 - H_norm, 0.5, 0.95)), 2)
        deblock_balance = H_norm - Phi
        auto_deblock_offset = int(np.clip(np.round(6.0 * deblock_balance), -6, 6))
        auto_deblock_str = f"{auto_deblock_offset},{auto_deblock_offset}"
        auto_aq_strength = round(float(np.clip(3.0 * Phi * (1.0 - H_norm), 0.0, 3.0)), 2)
        auto_tskip = 1 if H_norm < 0.35 else 0
        auto_rskip_edge = int(round(np.clip(1.0 + 9.0 * H_norm, 1.0, 100.0)))

        print(f"\n--- Calculated Spatial-Temporal Metrics ---")
        print(f"  > Visual Contrast Ratio (Phi):        {high_precision_float(Phi)}")
        print(f"  > Macro-Change Probability (Omega):   {high_precision_float(Omega)}")
        print(f"  > Otsu Separability (S_val):          {high_precision_float(S_val)}")
        print(f"  > Normalized Change Entropy (H_norm): {high_precision_float(H_norm)}")

        print(f"\n--- Physically Calibrated x265 Parameters ---")
        print(f"  > Max Reference Frames (ref):         {auto_ref}")
        print(f"  > Max Consecutive B-Frames:           {auto_bframes}")
        print(f"  > B-Frame Bias (Odds Ratio):          {auto_b_bias}")
        print(f"  > Scenecut Sensitivity:               {auto_scenecut}")
        print(f"  > Psychovisual RD strength:           {high_precision_float(auto_psy_rd)}")
        print(f"  > Psychovisual RDOQ trellis strength: {high_precision_float(auto_psy_rdoq)}")
        print(f"  > Rate Control qcomp (Curve Comp):    {high_precision_float(auto_qcomp)}")
        print(f"  > Deblock Filter Configuration:       deblock={auto_deblock_str}")
        print(f"  > AQ Strength (Bias toward flats):    {high_precision_float(auto_aq_strength)}")
        print(f"  > Transform Skip (Lossless 4x4):      tskip={auto_tskip}")
        print(f"  > Recursion Skip Edge Threshold:      rskip-edge-threshold={high_precision_float(auto_rskip_edge)}")
        print(f"----------------------------------------------------\n")

        x265_cfg = {
            "Profile": f"ref={auto_ref}",
            "Analysis": f"rd=6:rskip=2:rskip-edge-threshold={auto_rskip_edge}:rdoq-level=1:tu-intra-depth=4:tu-inter-depth=4:tskip={auto_tskip}",
            "Motion": "max-merge=5:subme=7:weightb=1:hme=1:hme-search=star,star,star:analyze-src-pics=1",
            "Intra": "strong-intra-smoothing=0:constrained-intra=1",
            "Psy": f"psy-rd={high_precision_float(auto_psy_rd)}:psy-rdoq={high_precision_float(auto_psy_rdoq)}",
            "GOP": f"open-gop=0:keyint=-1:min-keyint=0:scenecut={auto_scenecut}:hist-scenecut=1:rc-lookahead={auto_lookahead}:b-adapt=2:bframes={auto_bframes}:bframe-bias={auto_b_bias}:fades=1",
            "RC": f"crf=30:aq-mode=4:aq-strength={high_precision_float(auto_aq_strength)}:qp-adaptation-range=6.0:aq-motion=1:qg-size=8:qcomp={high_precision_float(auto_qcomp)}",
            "Filters": f"deblock={auto_deblock_str}:sao=0",
            "VUI": "repeat-headers=1:opt-qp-pps=1:opt-ref-list-length-pps=1:opt-cu-delta-qp=1"
        }

        print(f"\nPhase 4/4: Master Encode [x265 Veryslow 8-bit]")

        v_filter = f"mpdecimate=hi={safe_float(best_hi)}:lo={safe_float(best_lo)}:frac={safe_float(mpdecimate_frac)},setpts=PTS-STARTPTS"

        audio_inputs = ['-i', temp_audio_file] if has_audio else []
        audio_outputs = ['-c:a', 'libopus', '-b:a', AUDIO_BITRATE, '-ac', '1', '-vbr', 'on'] if has_audio else ['-an']

        ffmpeg_cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-fflags', '+genpts',
            '-f', 'rawvideo', '-pix_fmt', input_pix_fmt, '-s', resolution, '-r', safe_float(fps),
            '-i', 'pipe:0'
        ] + audio_inputs + [
            '-vf', v_filter,
            '-c:v', 'libx265', '-preset', 'veryslow', '-x265-params', ":".join(x265_cfg.values()),
            '-profile:v', 'main', '-pix_fmt', output_pix_fmt, '-color_range', 'pc', '-colorspace', 'bt709',
            '-color_primaries', 'bt709', '-color_trc', 'iec61966-2-1',
        ] + audio_outputs + [
            '-fps_mode', 'vfr',
            '-movflags', '+faststart', output_file
        ]

        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=sys.stderr)
        buffered_stdin = io.BufferedWriter(process.stdin, buffer_size=io_buffer_size)

        start_sec = float(START_TIME) if START_TIME else 0.0
        first_start = merged[0][0] if merged else start_sec
        starts = [s for s, e in merged]

        total_remaining_sec = max(0.0, total_sec - start_sec)
        v_total = int(total_remaining_sec * fps)

        h_blocks, w_blocks = height // 8, width // 8
        total_blocks = h_blocks * w_blocks
        best_n_blocks = int(round(mpdecimate_frac * total_blocks))

        run_length = 0
        best_pixels = None
        best_score = float('inf')
        ref_frame_luma = None

        def get_cleanliness_score(luma_arr):
            luma_arr = luma_arr[::4, ::4]
            diff_h = np.abs(luma_arr[1:, :] - luma_arr[:-1, :])
            diff_w = np.abs(luma_arr[:, 1:] - luma_arr[:, :-1])

            t_noise = max(1.0, best_lo / 64.0)
            t_edge = 6.0 * t_noise
            alpha = 0.1 * Phi * (1.0 - H_norm)

            noise_h_mask = (diff_h > 0) & (diff_h <= t_noise)
            noise_w_mask = (diff_w > 0) & (diff_w <= t_noise)
            noise_energy = (np.sum(diff_h[noise_h_mask]) + np.sum(diff_w[noise_w_mask])) / luma_arr.size

            edge_h_mask = (diff_h >= t_edge)
            edge_w_mask = (diff_w >= t_edge)
            edge_energy = (np.sum(diff_h[edge_h_mask]) + np.sum(diff_w[edge_w_mask])) / luma_arr.size

            return noise_energy - (alpha * edge_energy)

        with tqdm(total=v_total, unit="frame", desc="Encoding") as pbar:
            container.seek(int(first_start / float(v_stream.time_base)), stream=v_stream)

            for frame in container.decode(v_stream):
                ts = float(frame.pts * v_stream.time_base) if frame.pts is not None else 0.0
                if ts < first_start: continue

                idx = bisect.bisect_right(starts, ts)
                if idx > 0 and ts <= merged[idx - 1][1]:
                    frame_pixels = frame.to_ndarray(format=pix_fmt)
                    arr_luma = extract_luma_from_ndarray(frame_pixels, height, width)

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

                    score = get_cleanliness_score(arr_luma)

                    if is_different:
                        if run_length > 0 and best_pixels is not None:
                            for _ in range(run_length):
                                buffered_stdin.write(memoryview(best_pixels))

                        run_length = 1
                        best_pixels = np.ascontiguousarray(frame_pixels)
                        best_score = score
                        ref_frame_luma = arr_luma
                    else:
                        run_length += 1
                        if score < best_score:
                            best_pixels = np.ascontiguousarray(frame_pixels)
                            best_score = score

                pbar.update(1)
                if pbar.format_dict['rate']:
                    pbar.set_postfix_str("Speed: " + str(safe_float(pbar.format_dict['rate'] / fps)) + "x")

            if run_length > 0 and best_pixels is not None:
                for _ in range(run_length):
                    buffered_stdin.write(memoryview(best_pixels))

        buffered_stdin.close()
        process.wait()
        container.close()

    finally:
        # Clean up temporary WAV track safely from disk
        if 'temp_audio_file' in locals() and os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
                print(f"  > Cleaned up temporary audio track: {temp_audio_file}")
            except Exception as e:
                print(f"Warning: Could not remove temporary audio track {temp_audio_file}: {e}")


if __name__ == "__main__":
    run_pipeline()