# adaptive-screen-content-video-compressor

A physical, data-driven screen recording archival pipeline.

---

## Design Philosophy: Standard-Compliant Screen Content Emulation

Unlike traditional encoders that rely on hardcoded presets or arbitrary static boundaries, this pipeline dynamically analyzes video frames, extracts physical and statistical properties, and maps them onto standard-compliant H.265 (HEVC) parameters. The entire encoding process is self-optimizing and continuously variable, letting the underlying mathematics of the video signal drive the output parameters without heuristics.

---

## Technical Architecture & Coding-Theoretic Mapping

Instead of arbitrary thresholds or hard clipping cliffs, the pipeline maps continuous spatial-temporal metrics ($H_{\text{norm}}$, $\Omega$, $\Phi$, $S_{\text{val}}$) directly onto standard H.265 (x265) parameters using smooth, self-bounding mathematical functions:

### 1. Reference Picture Buffer Boundaries (`ref`)
The Decoded Picture Buffer (DPB) required for standard decoding is strictly constrained by HEVC specifications:
$$\text{DPB} = \text{ref} + \text{reorder depth} \le 8$$
Where the reorder depth is dynamically computed based on the B-frame structure (3 frames for $\text{bframes} \ge 4$ under a B-pyramid, 2 frames for $\text{bframes} < 4$) to prevent playback failures or hardware decoder memory overflow on consumer chipsets. Max safe reference ceiling is $\text{max safe ref} = 8 - \text{reorder depth}$.

To allocate reference frames continuously without arbitrary clipping boundaries, we map the relative sequence stability $u = \frac{S_{\text{val}}}{\Omega + \epsilon}$ onto $[1, \text{max safe ref}]$ using an algebraic growth curve:
$$\text{ref} = 1 + (\text{max safe ref} - 1) \cdot \frac{u}{u + 1}$$
As $\Omega \to 0$ (highly static screen state), references approach $\text{max safe ref}$ asymptotically. As $\Omega \to 1$ (chaotic motion), references decay smoothly to $1$.

### 2. B-Frame Allocations and Slicetype Bias
* **Consecutive B-Frames (`bframes`):** B-frame structure is scaled continuously between $4$ and $16$ based on the joint spatial-temporal complexity metric $H_{\text{norm}} \cdot \Omega$:
  $$\text{bframes} = 4 + 12 \cdot (1.0 - \Omega \cdot H_{\text{norm}})$$
* **Slicetype Bias (`bframe-bias`):** Preference for B-frame decision is mapped continuously onto the standard $[-100, 100]$ range via a hyperbolic tangent centered on a neutral 50% transition rate:
  $$\text{bframe-bias} = 100 \cdot \tanh(1.0 - 2\Omega)$$
  This guarantees that the bias safely scales between $+100$ (completely static) and $-100$ (constant temporal changes) without hitting arbitrary clamping cliffs.

### 3. Loop Filter Deblocking Offsets (`deblock`)
Applying positive deblocking values to screen content blurs sharp vector text lines, destroying readability. To enforce negative/neutral deblocking, we map the spatial-temporal balance $x = H_{\text{norm}} - \Phi$ continuously onto the safe window $(-6, 0)$ via a shifted hyperbolic tangent:
$$\text{deblock} = -3 \cdot \left(1.0 - \tanh(2x)\right)$$
* As $x \to -\infty$ (high contrast, low entropy $\implies$ sharp text), deblocking approaches $-6$ (maximum sharpness).
* As $x \to +\infty$ (low contrast, high entropy $\implies$ noisy textures), deblocking approaches $0$ (neutral).

### 4. Contrast-Preserving Adaptive Quantization (`aq-strength`)
Scaling spatial AQ strength to the standard ceiling of $3.0$ causes severe ringing noise around text boundaries. We continuously map the contrast-sparsity product onto the safe screen-content range of $[0.5, 1.5]$ using a hyperbolic tangent:
$$\text{aq-strength} = 0.5 + \tanh\left(\Phi \cdot (1.0 - H_{\text{norm}})\right)$$
This naturally scales up to $1.5$ to protect solid backgrounds from banding only when the content is highly static and contrasty, and decays smoothly to $0.5$ for high-entropy content to prevent mosquito noise.

### 5. Psychovisual RD and RDOQ (`psy-rd`, `psy-rdoq`)
High psychovisual RD values inject artificial high-frequency texture noise into flat UI elements. We continuously suppress these tools on high-contrast vector boundaries ($\Phi \to 1$) and scale them up on soft camera textures ($\Phi \to 0$) using an exponential decay curve:
$$\text{psy-rd} = 1.5 \cdot e^{-3\Phi}$$
$$\text{psy-rdoq} = 1.0 \cdot e^{-3\Phi}$$

### 6. Transform Skip (`tskip`)
Rather than using arbitrary binary threshold switches, the pipeline globally enables transform skip (`tskip = 1`) for screen content. This allows the HEVC rate-distortion optimization loop to natively evaluate and skip block-level discrete cosine transforms (DCT) where beneficial.

### 7. Scenecut Sensitivity (`scenecut`)
Otsu separability ($S_{\text{val}}$) is continuously mapped onto the safe search window of $(10, 90)$ via a logistic sigmoid centered at $0.5$:
$$\text{scenecut} = 10 + \frac{80}{1.0 + e^{-4(S_{\text{val}} - 0.5)}}$$

---

## Mathematical Core & Algorithms

### 1. Three-Class Otsu Thresholding Sweep
To isolate digital rendering noise without interfering with structural updates, we partition the absolute frame difference histogram into three mathematically distinct regions: Class 0 (Background Noise), Class 1 (Micro-Changes, such as text reveals and cursor updates), and Class 2 (Macro-Changes, such as slide wipes).

* **Noise Ceiling Invariant (`best_lo`):** Instead of mapping the noise ceiling to the high-contrast threshold $t_2$, we map it strictly to the lower threshold **$t_1$** (which separates Class 0 from Class 1):
  $$\text{best lo} = \max(\text{MIN PHYSICAL LO}, \text{t1})$$
  This invariant ensures that Class 1 micro-changes are correctly preserved in the active frame-keep count, eliminating the need for presentation-specific branch logic.
* **Macro-Boundary (`best_hi`):** Locked strictly to the physical maximum of an 8x8 block ($\text{MAX PHYSICAL HI} = 16320.0$).

### 2. AR(2) Infinite-Limit Parameter Extrapolation
Modeled as a second-order autoregressive process to project parameter behavior to its infinite-limit fixed point:
$$y_n = A_1 y_{n-1} + A_2 y_{n-2} + c + \epsilon$$
Evaluates roots of the characteristic equation $z^2 - A_1 z - A_2 = 0$. If both roots lie strictly inside the complex unit circle, the system is deemed stable, and parameters converge to $y^* = c / (1 - A_1 - A_2)$.

### 3. Algorithmic Complexity Crossover (Fast Threshold Counting)
Evaluates performance boundaries: Broadcast ($N \times T$) vs Suffix-Sum Bincount ($N + M$). The complexity crossover limit is calculated dynamically at runtime based on the actual size of the threshold array ($T$), ensuring optimal execution paths with zero hardcoded bottlenecks:
$$\text{crossover limit} = \frac{\text{HIST SIZE}}{T - 1}$$

### 4. Gradient-Based Visual Cleanliness Scoring
Evaluates texture and noise energy on a decimation pass to preserve visual cleanliness:
$$\text{Score} = \text{Noise Energy} - (\alpha \times \text{Edge Energy})$$
High-energy noise below $t_{\text{noise}} \le \text{best lo}/64$, high-frequency edges above $t_{\text{edge}} \ge 6 \times t_{\text{noise}}$.

---

## Speech-Driven Temporal Segmentation & Audio Workflow

### 1. Feature Extraction
Decodes the audio track and extracts root-mean-square (RMS) energy, zero-crossing rate (ZCR), and spectral energy in the human speech band ($300\text{ Hz} - 4000\text{ Hz}$).

### 2. Adaptive Gate
Otsu sweeps over audio features to isolate active speech from ambient noise.

### 3. Early Audio Export (Phase 2.5)
To optimize resource allocation, the trimmed WAV track is generated immediately after Phase 2 (Interval Generation) finishes. This decouples the audio-pipeline tasks, running disk-heavy audio exports early before the CPU is saturated by parallel visual sweep processes in Phase 3.

### 4. Preserved Cleanup Ownership
Deletion of the intermediate WAV file is omitted from `archiver_core.py` and delegated to the orchestration layer (`archiver_pipeline.ps1`), preventing premature file removal before final `.mkv` to `.mp4` container remuxing.
