# Adaptive Screen Content Video Compressor

An automated, data-driven screen recording archival pipeline. It dynamically analyzes video frames and speech activity to perform content-adaptive, high-efficiency x265 compression while retaining structural readability.

The pipeline operates by modeling the physical and statistical properties of active video segments, adjusting video encoding parameters to stay within the limits of standard HEVC Main Profile decoders.

---

## Design Philosophy: Standard-Compliant Screen Content Emulation

While H.265 contains a dedicated Screen Content Coding (SCC) extension, standard consumer hardware decoders often do not support it, and custom builds of x265 are frequently required to enable it. 

This project emulates the visual benefits of SCC while remaining strictly compliant with the standard H.265 Main Profile. By evaluating spatial-temporal change entropy, the pipeline dynamically maps standard parameters to match the unique characteristics of screen captures (e.g., sharp text, flat background geometry, low temporal motion).

---

## Technical Architecture & Coding-Theoretic Mapping

Every visual and temporal encoding parameter in this pipeline is dynamically calculated from the statistical properties of active audiovisual segments:

### 1. Reference Picture Buffer Boundaries (`ref`)
- **Limitation:** In the standard HEVC Main Profile, the Decoded Picture Buffer (DPB) is capped at 8 references. Because B-pyramid remains active, the pipeline limits the maximum L0 reference frames to 6. This restriction prevents playback failures or hardware decoder memory overflow on consumer chipsets.
- **Dynamic Derivation:** The reference count scales inversely with the scene transition rate ($\Omega$), mapping the geometric decay of static blocks onto a range of `[1, 6]`.

### 2. B-Frame Allocations and Slicetype Bias
- **Maximum B-Frames (`bframes`):** B-frames are scaled from a structural ceiling of 16 down to a floor of 4 based on combined change rate and entropy.
- **Slicetype Bias (`bframe-bias`):** Employs a symmetric log-odds (logit) stability model to project temporal stability. The log-odds value is mapped via a hyperbolic tangent function ($\tanh$) and scaled to fit the `[-90, 100]` x265 CLI range to prevent driver warning states.

### 3. Loop Filter Deblocking Offsets (`deblock`)
- **Method:** Balances vector text sharpness against dither-noise smoothing by applying a linear projection of change entropy (which dictates smoothing needs) against visual contrast (which dictates sharpness needs).
- **Mapping:** Maps the resulting balance across the physical range of `[-6, 6]`.

### 4. Contrast-Preserving Adaptive Quantization (`aq-strength`)
- **Method:** Maps spatial AQ strength dynamically to protect flat, clean vector backgrounds from banding and blockiness while preventing the encoder from wasting bitrate on high-frequency, randomized screen-capture noise.
- **Mapping:** Scales across the standard `[0.0, 3.0]` range.

---

## Mathematical Core & Algorithms

The pipeline utilizes several analytical mathematical models to evaluate and compress visual screen content:

### 1. Three-Class Otsu Thresholding Sweep
Traditional Otsu thresholding splits a probability density function (PDF) into two classes. This pipeline implements a **3-class Otsu sweep** that maximizes the between-class variance to separate frame changes into three mathematically distinct regions:
- **Class 0 (Background Noise):** Captures sub-pixel luma fluctuations, dither noise, and sensor dither.
- **Class 1 (Micro-Changes):** Isolates subtle UI changes such as cursor blinks, hovering indicators, or microphone activity.
- **Class 2 (Macro-Changes):** Identifies actual structural updates (e.g., text typing, screen sliding, window transitions).

This classification provides the structural probability ($\Omega$) and contrast ratio ($\Phi$) metrics used to tune the encoder parameters.

### 2. AR(2) Infinite-Limit Parameter Extrapolation
During calibration sweeps, parameters such as the low block-sum threshold (`lo`) and minimum active block counts (`n`) can oscillate between iterations. To ensure mathematical convergence, the calibration engine monitors history states and projects parameter behavior to its infinite-limit fixed point using a second-order auto-regressive model with drift:

$$y_n = A_1 y_{n-1} + A_2 y_{n-2} + c + \epsilon$$

- **Stability Check:** The characteristic equation roots (`z^2 - A1*z - A2 = 0`) are evaluated. If both roots lie strictly inside the complex unit circle, the system is deemed stable (handling both monotonic and damped-oscillating convergence).
- **Exit Strategy:** The system early-exits when the projected infinite-limit fixed point rounds to the same integer values on consecutive sweeps, saving CPU cycles.

### 3. Algorithmic Complexity Crossover (Fast Threshold Counting)
When evaluating changes across 8x8 pixel blocks, the engine must count how many blocks exceed various change thresholds. The pipeline implements a complexity-theory crossover optimization that switches between two mathematical models:
1. **Broadcast Method ($N \times T$ complexity):** Evaluates $N$ active blocks against $T$ discrete thresholds.
2. **Suffix-Sum Bincount Method ($N + M$ complexity):** Populates a deterministic bincount of size $M$ (16,321 possible block sums) and computes a suffix sum.

The crossover point occurs exactly where the computational densities intersect:

$$N = \frac{M}{T - 1}$$

This solves to exactly **64 blocks** (`CROSSOVER_LIMIT`). If fewer than 64 blocks are active, the broadcast model runs; otherwise, the suffix-sum method is utilized, avoiding redundant comparisons.

### 4. Gradient-Based Visual Cleanliness Scoring
When duplicating repeated visual states across decimation intervals, the pipeline evaluates each candidate frame's spatial gradients to select the cleanest frame to write.
- **Noise Energy:** Sums absolute horizontal and vertical pixel-to-pixel differences that fall below a dynamically calibrated noise threshold (`t_noise <= best_lo / 64`).
- **Edge Energy:** Sums gradient changes that fall above a dynamic edge threshold (`t_edge >= 6 * t_noise`).
- **Objective Score:** Employs a regularized minimization objective:

  $$\text{Score} = \text{Noise Energy} - (\alpha \times \text{Edge Energy})$$

  The frame with the lowest score (representing minimal sub-pixel noise and maximum edge sharpness) is preserved.

---

## Speech-Driven Temporal Segmentation

To isolate and segment spoken portions of the audio track before video encoding:
1. **Feature Extraction:** Decodes the audio track and extracts root-mean-square (RMS) energy, zero-crossing rate (ZCR), and spectral energy in the human speech band ($300\text{ Hz} - 4000\text{ Hz}$).
2. **Adaptive Gate Determination:** Automatically establishes threshold limits using individual Otsu sweeps over the extracted audio feature profiles.
3. **Margin Padding:** Segments are padded with configurable margins (`MARGIN_SECS`) to prevent clipping conversational speech before merging adjacent intervals.

---

## Installation & Usage

### 1. Requirements
Ensure you have the following dependencies configured on your system:
- **Python:** 3.8 or higher.
- **FFmpeg:** Must be globally accessible via your system's path.
- **Dependencies:** Install via the provided `requirements.txt`:
  ```bash
  pip install -r requirements.txt
