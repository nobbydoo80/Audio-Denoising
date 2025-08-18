# Audio Restoration Strategy: Interview Denoising with Child Vocalization Removal

## Executive Summary
This document outlines a comprehensive, multi-stage approach for removing persistent high-pitched child vocalizations (screaming, crying, and other disruptive sounds) from the "Jock" interview recording. The strategy leverages the existing wavelet-based denoising implementation while incorporating advanced spectral processing, adaptive filtering, and AI-based source separation techniques.

## Available Resources
- **Primary Recording**: `jock_itntvw.wav` - Interview with subject "Jock" contaminated with child vocalizations
- **Noise Profile**: `evilhild_2.wav` - Isolated sample of unwanted child sounds for noise profiling
- **Existing Tools**: Python-based wavelet denoising implementation using PyWavelets

---

## Phase 1: Initial Analysis and Preparation

### 1.1 Audio File Assessment
```python
# Analyze both files for spectral characteristics
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Load files
interview_data, interview_rate = sf.read('jock_itntvw.wav')
noise_data, noise_rate = sf.read('evilhild_2.wav')

# Perform spectral analysis
frequencies_interview, times_interview, Sxx_interview = signal.spectrogram(
    interview_data, interview_rate, nperseg=2048
)
frequencies_noise, times_noise, Sxx_noise = signal.spectrogram(
    noise_data, noise_rate, nperseg=2048
)
```

**Key Parameters to Document:**
- Sample rate consistency (ensure both files match)
- Dynamic range analysis
- Frequency distribution of child vocalizations (typically 1-4 kHz for crying/screaming)
- Temporal patterns of noise intrusions
- Signal-to-Noise Ratio (SNR) estimation

### 1.2 Noise Profile Characterization
Using the isolated "Evil Child" sample:
- **Frequency Range**: Identify dominant frequencies (typically 800-4000 Hz for child vocalizations)
- **Harmonic Structure**: Map fundamental frequencies and harmonics
- **Temporal Envelope**: Analyze attack/decay characteristics
- **Spectral Centroid**: Calculate average frequency content weight

### 1.3 Pre-processing Setup
```python
# Normalize audio levels
interview_normalized = interview_data / np.max(np.abs(interview_data))
noise_normalized = noise_data / np.max(np.abs(noise_data))

# Create backup of original
sf.write('jock_interview_backup.wav', interview_data, interview_rate)
```

---

## Phase 2: Primary Noise Reduction Pipeline

### 2.1 Wavelet-Based Denoising (Existing Implementation)
Utilize the repository's wavelet transform approach with optimized parameters:

```python
from denoise import AudioDeNoise

# Stage 1: Initial wavelet denoising
denoiser = AudioDeNoise(inputFile="jock_itntvw.wav")

# Generate noise profile from isolated sample
denoiser.generateNoiseProfile(noiseFile="evilhild_2.wav")

# Apply denoising with modified parameters
denoiser.deNoise(outputFile="jock_stage1_wavelet.wav")
```

**Recommended Wavelet Parameters:**
- **Wavelet Family**: `db4` (Daubechies 4) - good for speech
- **Decomposition Level**: 4-6 levels for comprehensive frequency coverage
- **Thresholding Method**: Soft thresholding with VisuShrink
- **Threshold Multiplier**: Start with 0.8x standard threshold for less aggressive removal

### 2.2 Spectral Subtraction Enhancement
Apply frequency-domain noise reduction using the noise profile:

```python
def spectral_subtraction(signal_fft, noise_profile_fft, alpha=2.0, beta=0.1):
    """
    Enhanced spectral subtraction with over-subtraction factor
    
    Parameters:
    - alpha: Over-subtraction factor (1.5-3.0 for aggressive removal)
    - beta: Spectral floor parameter (0.05-0.2 to prevent over-suppression)
    """
    magnitude = np.abs(signal_fft)
    phase = np.angle(signal_fft)
    noise_magnitude = np.abs(noise_profile_fft)
    
    # Apply subtraction with floor
    clean_magnitude = magnitude - alpha * noise_magnitude
    clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)
    
    # Reconstruct signal
    return clean_magnitude * np.exp(1j * phase)
```

**Recommended Parameters:**
- **FFT Size**: 2048 samples (46ms @ 44.1kHz)
- **Overlap**: 75% for smooth reconstruction
- **Alpha (over-subtraction)**: 2.0-2.5 for child vocalizations
- **Beta (spectral floor)**: 0.1 to preserve speech naturalness

### 2.3 Adaptive Wiener Filtering
Implement frequency-dependent gain adjustment:

```python
def wiener_filter(noisy_spectrum, noise_power_spectrum, speech_power_estimate):
    """
    Adaptive Wiener filter for speech enhancement
    """
    snr_estimate = speech_power_estimate / (noise_power_spectrum + 1e-10)
    wiener_gain = snr_estimate / (1 + snr_estimate)
    
    # Apply frequency-dependent smoothing
    wiener_gain = signal.medfilt(wiener_gain, kernel_size=5)
    
    return noisy_spectrum * wiener_gain
```

**Configuration:**
- **Noise Power Estimation**: Use first 500ms of "Evil Child" sample
- **Speech Power Estimation**: Use voice activity detection (VAD)
- **Smoothing Window**: 5-bin median filter for gain function
- **Update Rate**: Every 20ms for adaptive tracking

---

## Phase 3: Advanced Processing Techniques

### 3.1 Harmonic-Percussive Source Separation (HPSS)
Separate speech (harmonic) from transient noise:

```python
import librosa

# Apply HPSS to isolate speech components
harmonic, percussive = librosa.effects.hpss(
    interview_data,
    margin=(1.0, 5.0),  # Favor harmonic for speech
    kernel_size=31
)
```

**Parameters:**
- **Margin**: (1.0, 5.0) - Conservative harmonic preservation
- **Kernel Size**: 31 frames for temporal smoothing
- **Power**: 2.0 for L2-norm separation

### 3.2 AI-Based Source Separation (Optional Advanced Stage)
For persistent artifacts, consider deep learning approaches:

**Recommended Models:**
1. **Facebook Demucs**: Real-time audio source separation
   ```bash
   pip install demucs
   python -m demucs --two-stems=vocals jock_itntvw.wav
   ```

2. **Spleeter by Deezer**: Pre-trained source separation
   ```bash
   pip install spleeter
   spleeter separate -o output -p spleeter:2stems jock_itntvw.wav
   ```

3. **OpenUnmix**: Open-source neural source separation
   ```python
   import openunmix
   separator = openunmix.umxhq()
   estimates = separator(audio_tensor)
   ```

### 3.3 Frequency-Specific Noise Gating
Target child vocalization frequency bands:

```python
def frequency_gate(signal_fft, gate_frequencies, threshold_db=-40):
    """
    Apply gating to specific frequency bands
    
    Gate Frequencies for Child Vocalizations:
    - Crying fundamental: 300-600 Hz
    - Screaming: 1000-3000 Hz
    - Harmonics: 2000-4000 Hz
    """
    magnitude = np.abs(signal_fft)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    for freq_range in gate_frequencies:
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        gate_mask = magnitude_db[freq_mask] < threshold_db
        magnitude[freq_mask][gate_mask] *= 0.1  # Aggressive attenuation
    
    return magnitude * np.exp(1j * np.angle(signal_fft))
```

**Recommended Gate Bands:**
- 300-600 Hz: Crying fundamental (threshold: -35 dB)
- 1000-3000 Hz: Screaming range (threshold: -40 dB)
- 3000-4000 Hz: High harmonics (threshold: -45 dB)

---

## Phase 4: Quality Control and Artifact Management

### 4.1 Musical Noise Suppression
Address spectral subtraction artifacts:

```python
def musical_noise_suppression(spectrum, smoothing_factor=0.98):
    """
    Temporal smoothing to reduce musical noise
    """
    # Apply recursive averaging
    smoothed = spectrum.copy()
    for i in range(1, len(spectrum)):
        smoothed[i] = smoothing_factor * smoothed[i-1] + \
                     (1 - smoothing_factor) * spectrum[i]
    return smoothed
```

### 4.2 Pumping Effect Mitigation
Prevent amplitude modulation artifacts:

```python
def anti_pumping_filter(gain_curve, attack_time=0.005, release_time=0.05):
    """
    Smooth gain transitions to prevent pumping
    
    Parameters:
    - attack_time: 5ms for quick response to speech
    - release_time: 50ms for smooth noise suppression
    """
    sample_rate = 44100
    attack_samples = int(attack_time * sample_rate)
    release_samples = int(release_time * sample_rate)
    
    smoothed_gain = np.zeros_like(gain_curve)
    
    for i in range(len(gain_curve)):
        if gain_curve[i] > smoothed_gain[i-1]:
            # Attack (fast)
            alpha = 1.0 - np.exp(-1.0 / attack_samples)
        else:
            # Release (slow)
            alpha = 1.0 - np.exp(-1.0 / release_samples)
        
        smoothed_gain[i] = alpha * gain_curve[i] + (1 - alpha) * smoothed_gain[i-1]
    
    return smoothed_gain
```

### 4.3 Frequency Masking Prevention
Preserve speech intelligibility:

```python
def preserve_formants(processed_spectrum, original_spectrum, formant_frequencies):
    """
    Protect speech formants from over-processing
    
    Typical Formant Ranges:
    - F1: 300-900 Hz
    - F2: 900-2500 Hz
    - F3: 2500-3500 Hz
    """
    protected_spectrum = processed_spectrum.copy()
    
    for f_range in formant_frequencies:
        freq_mask = (frequencies >= f_range[0]) & (frequencies <= f_range[1])
        # Blend processed and original in formant regions
        protected_spectrum[freq_mask] = 0.7 * processed_spectrum[freq_mask] + \
                                       0.3 * original_spectrum[freq_mask]
    
    return protected_spectrum
```

---

## Phase 5: Final Mastering and Post-Processing

### 5.1 Multi-band Compression
Balance frequency ranges post-denoising:

```python
from scipy.signal import butter, sosfilt

def multiband_compress(audio, sample_rate=44100):
    """
    4-band compression for final polish
    """
    bands = [
        (20, 250),      # Sub-bass/bass
        (250, 1000),    # Low-mids
        (1000, 4000),   # High-mids (critical for speech)
        (4000, 20000)   # Highs
    ]
    
    compressed_bands = []
    for low, high in bands:
        sos = butter(4, [low, high], btype='band', fs=sample_rate, output='sos')
        band_signal = sosfilt(sos, audio)
        
        # Apply band-specific compression
        if low == 1000:  # Speech-critical band
            ratio = 2:1
            threshold = -20
        else:
            ratio = 3:1
            threshold = -25
            
        compressed_bands.append(compress(band_signal, ratio, threshold))
    
    return np.sum(compressed_bands, axis=0)
```

### 5.2 Adaptive EQ Restoration
Compensate for frequency loss:

```python
def adaptive_eq(processed_audio, reference_spectrum):
    """
    Match spectral envelope to reference
    """
    # Calculate spectral tilt correction
    processed_spectrum = np.fft.rfft(processed_audio)
    
    # Design compensation filter
    compensation_curve = reference_spectrum / (np.abs(processed_spectrum) + 1e-10)
    compensation_curve = np.clip(compensation_curve, 0.5, 2.0)  # Limit boost/cut
    
    # Apply smooth EQ curve
    from scipy.ndimage import gaussian_filter1d
    smooth_curve = gaussian_filter1d(compensation_curve, sigma=5)
    
    return np.fft.irfft(processed_spectrum * smooth_curve)
```

### 5.3 Final Limiting and Normalization
```python
def final_master(audio, target_lufs=-16):
    """
    Final stage mastering
    """
    # Peak limiting
    limited = np.tanh(audio * 0.9) / 0.9  # Soft saturation limiter
    
    # LUFS normalization
    import pyloudnorm as pyln
    meter = pyln.Meter(44100)
    loudness = meter.integrated_loudness(limited)
    normalized = pyln.normalize.loudness(limited, loudness, target_lufs)
    
    return normalized
```

---

## Phase 6: Quality Assurance Checkpoints

### 6.1 Objective Metrics
Monitor these metrics at each stage:

| Metric | Target Value | Measurement Tool |
|--------|-------------|------------------|
| SNR Improvement | >15 dB | `10 * log10(signal_power/noise_power)` |
| PESQ Score | >3.5 | `python-pesq` library |
| STOI (Intelligibility) | >0.85 | `pystoi` library |
| Spectral Distortion | <5 dB | RMS difference in spectrum |
| Crest Factor | 12-18 dB | Peak-to-RMS ratio |

### 6.2 Subjective Evaluation Checklist
- [ ] Speech clarity and intelligibility maintained
- [ ] No audible musical noise artifacts
- [ ] Absence of pumping/breathing effects
- [ ] Natural voice timbre preserved
- [ ] Child vocalizations effectively suppressed
- [ ] No frequency holes or unnatural gaps
- [ ] Consistent noise floor throughout

### 6.3 A/B Testing Protocol
1. Create multiple versions with different parameter sets
2. Perform blind listening tests on 30-second segments
3. Document problem areas requiring manual intervention
4. Note timestamps of residual artifacts for targeted processing

---

## Fallback Strategies for Challenging Sections

### Strategy A: Manual Spectral Editing
For sections with severe overlap between speech and noise:
1. Use **Audacity** or **iZotope RX** for visual spectral editing
2. Manually paint out noise frequencies while preserving speech
3. Apply spectral repair to fill gaps

### Strategy B: Parallel Processing
1. Create two processing chains:
   - Chain A: Aggressive noise removal (may damage speech)
   - Chain B: Conservative processing (may retain some noise)
2. Use envelope follower to blend between chains based on VAD

### Strategy C: Segment-Based Processing
```python
def segment_based_processing(audio, segment_duration=1.0):
    """
    Apply different processing based on segment characteristics
    """
    segments = []
    segment_samples = int(segment_duration * sample_rate)
    
    for i in range(0, len(audio), segment_samples):
        segment = audio[i:i+segment_samples]
        
        # Analyze segment
        has_speech = detect_voice_activity(segment)
        noise_level = estimate_noise_level(segment)
        
        # Apply appropriate processing
        if has_speech and noise_level > threshold:
            processed = aggressive_denoise(segment)
        elif has_speech:
            processed = moderate_denoise(segment)
        else:
            processed = noise_gate(segment)
            
        segments.append(processed)
    
    return np.concatenate(segments)
```

---

## Implementation Workflow

### Step-by-Step Execution Order:

1. **Initial Setup** (5 minutes)
   ```bash
   python analyze_audio.py --input jock_itntvw.wav --noise evilhild_2.wav
   ```

2. **Stage 1: Wavelet Denoising** (2-3 minutes)
   ```python
   python denoise.py --mode wavelet --aggressive 0.8
   ```

3. **Stage 2: Spectral Processing** (3-5 minutes)
   ```python
   python spectral_denoise.py --method subtraction --alpha 2.0
   ```

4. **Stage 3: Adaptive Filtering** (2-3 minutes)
   ```python
   python adaptive_filter.py --type wiener --vad enabled
   ```

5. **Stage 4: Source Separation** (5-10 minutes if using AI)
   ```bash
   python ai_separation.py --model demucs --stems vocals
   ```

6. **Stage 5: Artifact Removal** (2-3 minutes)
   ```python
   python remove_artifacts.py --musical-noise --pumping
   ```

7. **Stage 6: Final Master** (2 minutes)
   ```python
   python master_audio.py --lufs -16 --limiter soft
   ```

8. **Quality Check** (5 minutes)
   ```python
   python evaluate_quality.py --metrics all --reference original
   ```

### Total Processing Time: 25-35 minutes

---

## Software Tool Recommendations

### Primary Tools (Open Source)
1. **Python Libraries**:
   - `librosa`: Audio analysis and processing
   - `scipy.signal`: Signal processing operations
   - `pydub`: Audio manipulation
   - `pyroomacoustics`: Advanced acoustic processing
   - `noisereduce`: Stationary noise reduction

2. **Command Line Tools**:
   - `sox`: Swiss army knife of audio processing
   - `ffmpeg`: Audio format conversion and filtering

### Professional Tools (Commercial)
1. **iZotope RX 10**: Industry-standard restoration suite
   - Spectral De-noise module
   - Dialogue Isolate
   - De-rustle for handling movement noise

2. **Adobe Audition**: Comprehensive audio editing
   - Adaptive Noise Reduction
   - Sound Remover effect
   - Spectral Frequency Display

3. **Waves NS1**: Real-time noise suppressor
   - Simple one-knob operation
   - Minimal artifacts

---

## Critical Success Factors

1. **Preserve Dialogue Intelligibility**: Never sacrifice speech clarity for noise reduction
2. **Iterative Processing**: Apply multiple gentle passes rather than one aggressive pass
3. **Frequency-Selective Approach**: Target specific bands where child vocalizations dominate
4. **Temporal Consistency**: Maintain uniform processing to avoid jarring transitions
5. **Regular Checkpoints**: Save intermediate versions for comparison and rollback

---

## Troubleshooting Guide

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| Robotic/metallic voice | Over-aggressive spectral subtraction | Reduce alpha factor, increase spectral floor |
| Pumping/breathing | Rapid gain changes | Increase attack/release times, smooth gain curve |
| Musical noise | Isolated frequency peaks | Apply median filtering, increase FFT overlap |
| Muffled speech | Over-processing of formants | Protect 1-3 kHz range, reduce processing depth |
| Residual child noise | Insufficient noise profile | Extend noise sample, use adaptive thresholding |
| Frequency holes | Excessive gating | Reduce gate depth, use gradual attenuation |

---

## Conclusion

This comprehensive strategy provides multiple approaches to tackle the challenging task of removing child vocalizations from the interview recording. The multi-stage pipeline allows for incremental improvement while maintaining quality checkpoints to prevent over-processing. The combination of traditional DSP techniques with modern AI-based approaches offers flexibility to handle varying noise conditions throughout the recording.

Key to success is the iterative application of complementary techniques, careful parameter tuning based on the specific characteristics of the child vocalizations, and constant monitoring of speech quality metrics. The fallback strategies ensure that even the most challenging sections can be addressed through alternative methods.

Remember: The goal is not perfect silence but natural-sounding speech with minimized distractions. Some residual ambient noise may be preferable to heavily processed, artificial-sounding audio.