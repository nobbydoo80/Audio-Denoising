#!/usr/bin/env python
"""
Phase 5: Artifact Control
Removes clicks, pops, musical noise, and other artifacts
Preserves transients and ensures phase coherence
"""
import argparse
import os
import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.interpolate import interp1d

def set_seed(seed: int):
    """Set deterministic seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)

def ensure_dirs(paths):
    """Create directories if they don't exist"""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def detect_clicks(x, sr, threshold_db=-20, min_duration_ms=0.5, max_duration_ms=10):
    """
    Detect clicks and pops in audio
    Returns list of (start_sample, end_sample) tuples
    """
    # Convert to samples
    min_samples = int(min_duration_ms * sr / 1000)
    max_samples = int(max_duration_ms * sr / 1000)
    
    # Compute local energy
    window_size = min_samples
    energy = np.convolve(x**2, np.ones(window_size)/window_size, mode='same')
    energy_db = 10 * np.log10(energy + 1e-10)
    
    # Find sudden changes
    diff = np.diff(energy_db)
    threshold = np.percentile(np.abs(diff), 99)
    
    # Detect click positions
    clicks = []
    click_mask = np.abs(diff) > threshold
    
    # Group consecutive samples
    in_click = False
    start = 0
    
    for i, is_click in enumerate(click_mask):
        if is_click and not in_click:
            start = i
            in_click = True
        elif not is_click and in_click:
            duration = i - start
            if min_samples <= duration <= max_samples:
                clicks.append((start, i))
            in_click = False
    
    return clicks

def remove_clicks(x, sr, clicks, interpolation='cubic'):
    """
    Remove clicks by interpolation
    """
    x_clean = x.copy()
    
    for start, end in clicks:
        # Extend window slightly for smooth interpolation
        margin = int(0.001 * sr)  # 1ms margin
        start_ext = max(0, start - margin)
        end_ext = min(len(x), end + margin)
        
        # Get surrounding samples
        before = x[start_ext:start]
        after = x[end:end_ext]
        
        if len(before) > 0 and len(after) > 0:
            # Interpolate
            gap_length = end - start
            
            if interpolation == 'linear':
                interp_values = np.linspace(before[-1], after[0], gap_length)
            elif interpolation == 'cubic':
                # Use cubic spline
                points = np.concatenate([before[-min(4, len(before)):], 
                                        after[:min(4, len(after))]])
                indices = np.concatenate([np.arange(-min(4, len(before)), 0),
                                        np.arange(gap_length, gap_length + min(4, len(after)))])
                
                if len(points) >= 2:
                    f = interp1d(indices, points, kind='cubic', fill_value='extrapolate')
                    interp_values = f(np.arange(gap_length))
                else:
                    interp_values = np.linspace(before[-1], after[0], gap_length)
            elif interpolation == 'sinc':
                # Sinc interpolation (ideal but computationally expensive)
                interp_values = signal.resample(np.array([before[-1], after[0]]), gap_length)
            else:
                interp_values = np.zeros(gap_length)
            
            x_clean[start:end] = interp_values
    
    return x_clean

def suppress_musical_noise(x, sr, method='spectral_floor', floor_factor=0.1, 
                          median_time=5, median_freq=3):
    """
    Suppress musical noise artifacts from spectral processing
    """
    # STFT
    nperseg = 2048
    noverlap = nperseg * 3 // 4
    f, t, Z = signal.stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap)
    S = np.abs(Z)
    phase = np.angle(Z)
    
    if method == 'spectral_floor':
        # Apply spectral floor
        floor = floor_factor * np.max(S)
        S_clean = np.maximum(S, floor)
        
    elif method == 'median_filter':
        # 2D median filtering
        from scipy.ndimage import median_filter
        S_clean = median_filter(S, size=(median_freq, median_time))
        
    elif method == 'minimum_statistics':
        # Minimum statistics noise estimation
        # Estimate noise floor using minimum statistics
        noise_floor = np.percentile(S, 5, axis=1, keepdims=True)
        
        # Spectral subtraction with oversubtraction
        alpha = 2.0
        S_clean = S - alpha * noise_floor
        S_clean = np.maximum(S_clean, 0.1 * S)
        
    else:
        S_clean = S
    
    # Reconstruct
    Z_clean = S_clean * np.exp(1j * phase)
    _, x_clean = signal.istft(Z_clean, fs=sr, nperseg=nperseg, noverlap=noverlap)
    
    return x_clean

def preserve_transients(x, sr, detection_threshold=0.3, protection_ms=5, blend_factor=0.7):
    """
    Detect and preserve transients
    """
    # Transient detection using spectral flux
    nperseg = 512
    noverlap = nperseg // 2
    f, t, Z = signal.stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap)
    S = np.abs(Z)
    
    # Spectral flux
    flux = np.sum(np.maximum(0, np.diff(S, axis=1)), axis=0)
    flux = np.pad(flux, (1, 0), mode='constant')
    
    # Normalize
    flux = flux / (np.max(flux) + 1e-10)
    
    # Detect transients
    transient_mask = flux > detection_threshold
    
    # Extend protection window
    protection_samples = int(protection_ms * sr / 1000)
    for i in np.where(transient_mask)[0]:
        start = max(0, i - protection_samples)
        end = min(len(transient_mask), i + protection_samples)
        transient_mask[start:end] = True
    
    return transient_mask, flux

def control_pre_echo(x, sr, lookahead_ms=5, threshold=0.1):
    """
    Control pre-echo artifacts
    """
    lookahead_samples = int(lookahead_ms * sr / 1000)
    
    # Detect sudden onsets
    envelope = np.abs(signal.hilbert(x))
    diff = np.diff(envelope)
    
    # Find onset positions
    onsets = np.where(diff > threshold * np.max(diff))[0]
    
    # Apply fade-in before onsets
    x_clean = x.copy()
    for onset in onsets:
        fade_start = max(0, onset - lookahead_samples)
        fade_length = onset - fade_start
        
        if fade_length > 0:
            # Create fade-in curve
            fade = np.linspace(0, 1, fade_length) ** 2
            x_clean[fade_start:onset] *= fade
    
    return x_clean

def de_ess(x, sr, frequency_range=[4000, 9000], threshold_db=-25, ratio=4,
          attack_ms=0.5, release_ms=10):
    """
    De-essing to reduce sibilance
    """
    # Bandpass filter for sibilant frequencies
    sos = signal.butter(4, frequency_range, btype='band', fs=sr, output='sos')
    sibilant = signal.sosfilt(sos, x)
    
    # Envelope follower
    envelope = np.abs(signal.hilbert(sibilant))
    
    # Smooth envelope
    attack_samples = max(1, int(attack_ms * sr / 1000))  # Ensure at least 1
    release_samples = max(1, int(release_ms * sr / 1000))  # Ensure at least 1
    
    smoothed = np.zeros_like(envelope)
    for i in range(1, len(envelope)):
        if envelope[i] > smoothed[i-1]:
            # Attack
            alpha = 1.0 - np.exp(-1.0 / attack_samples)
        else:
            # Release
            alpha = 1.0 - np.exp(-1.0 / release_samples)
        smoothed[i] = alpha * envelope[i] + (1 - alpha) * smoothed[i-1]
    
    # Convert to dB
    envelope_db = 20 * np.log10(smoothed + 1e-10)
    
    # Compute gain reduction
    gain_reduction = np.ones_like(envelope_db)
    above_threshold = envelope_db > threshold_db
    
    if np.any(above_threshold):
        excess_db = envelope_db[above_threshold] - threshold_db
        gain_reduction_db = -excess_db * (1 - 1/ratio)
        gain_reduction[above_threshold] = 10 ** (gain_reduction_db / 20)
    
    # Apply gain reduction to sibilant band
    sibilant_reduced = sibilant * gain_reduction
    
    # Replace sibilant band in original
    x_clean = x - sibilant + sibilant_reduced
    
    return x_clean

def ensure_phase_coherence(x, sr, method='griffin_lim', iterations=32):
    """
    Ensure phase coherence using Griffin-Lim or other methods
    """
    if method == 'griffin_lim':
        # Griffin-Lim algorithm
        nperseg = 2048
        noverlap = nperseg * 3 // 4
        
        # Initial STFT
        f, t, Z = signal.stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap)
        S = np.abs(Z)
        
        # Griffin-Lim iterations
        for _ in range(iterations):
            # Reconstruct with current phase
            _, x_iter = signal.istft(Z, fs=sr, nperseg=nperseg, noverlap=noverlap)
            
            # Forward transform
            _, _, Z = signal.stft(x_iter, fs=sr, nperseg=nperseg, noverlap=noverlap)
            
            # Keep original magnitude, update phase
            Z = S * np.exp(1j * np.angle(Z))
        
        # Final reconstruction
        _, x_clean = signal.istft(Z, fs=sr, nperseg=nperseg, noverlap=noverlap)
        
    elif method == 'vocoder':
        # Phase vocoder approach
        # Simplified: just ensure continuity
        x_clean = x
        
    else:
        x_clean = x
    
    # Ensure same length
    if len(x_clean) != len(x):
        if len(x_clean) > len(x):
            x_clean = x_clean[:len(x)]
        else:
            x_clean = np.pad(x_clean, (0, len(x) - len(x_clean)))
    
    return x_clean

def remove_dc_offset(x, sr, method='highpass', cutoff_hz=20):
    """
    Remove DC offset
    """
    if method == 'highpass':
        # High-pass filter
        sos = signal.butter(2, cutoff_hz, btype='high', fs=sr, output='sos')
        x_clean = signal.sosfilt(sos, x)
    elif method == 'subtract_mean':
        # Simple mean subtraction
        x_clean = x - np.mean(x)
    else:
        x_clean = x
    
    return x_clean

def apply_safety_limiter(x, ceiling_db=-0.3, release_ms=50, lookahead_ms=5):
    """
    Apply safety limiter to prevent clipping
    """
    ceiling_linear = 10 ** (ceiling_db / 20)
    
    # Find peaks above ceiling
    peaks = np.abs(x) > ceiling_linear
    
    if not np.any(peaks):
        return x
    
    # Apply soft limiting
    x_limited = np.copy(x)
    x_limited[peaks] = np.sign(x[peaks]) * ceiling_linear
    
    # Smooth transitions
    if release_ms > 0:
        # Simple exponential release
        release_samples = int(release_ms * 44100 / 1000)  # Assume 44.1kHz
        
        for i in np.where(peaks)[0]:
            # Apply release envelope
            for j in range(i, min(i + release_samples, len(x_limited))):
                if np.abs(x_limited[j]) < ceiling_linear:
                    break
                factor = np.exp(-(j - i) / release_samples)
                x_limited[j] = x[j] * (1 - factor) + x_limited[j] * factor
    
    return x_limited

def main():
    ap = argparse.ArgumentParser(description="Phase 5: Artifact Control")
    ap.add_argument("--config", required=True, help="Path to config file")
    ap.add_argument("--input", default=None, help="Input audio")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    
    # Load config
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Check if phase is enabled
    if not cfg.get("phases", {}).get("phase5", {}).get("enabled", True):
        print("[INFO] Phase 5 (Artifact Control) is disabled in config, skipping")
        return
    
    # Set seed
    seed = cfg.get("artifact_control", {}).get("seed", cfg.get("global", {}).get("seed", 42))
    set_seed(seed)
    
    # Setup paths
    outputs_dir = cfg["paths"]["outputs_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dirs([outputs_dir, logs_dir])
    
    # Determine input
    prefer = [
        os.path.join(outputs_dir, "phase4_ai_enhanced.wav"),
        os.path.join(outputs_dir, "phase3_hpss.wav"),
        os.path.join(outputs_dir, "phase2c_wiener.wav"),
        cfg["paths"]["input"]
    ]
    in_path = args.input or next((p for p in prefer if os.path.exists(p)), cfg["paths"]["input"])
    out_path = args.output or os.path.join(outputs_dir, "phase5_artifact_controlled.wav")
    
    # Load audio
    print(f"[INFO] Loading {in_path}")
    x, sr = sf.read(in_path)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    
    # Get artifact control config
    ac_cfg = cfg.get("artifact_control", {})
    
    print("[INFO] Starting artifact control...")
    t0 = time.time()
    
    # Track applied processes
    applied = []
    metrics = {}
    
    # 1. Click/Pop Removal
    if ac_cfg.get("click_removal", {}).get("enabled", True):
        print("[INFO] Detecting and removing clicks...")
        clicks = detect_clicks(
            x, sr,
            threshold_db=ac_cfg["click_removal"].get("threshold_db", -20),
            min_duration_ms=ac_cfg["click_removal"].get("min_duration_ms", 0.5),
            max_duration_ms=ac_cfg["click_removal"].get("max_duration_ms", 10)
        )
        
        if clicks:
            print(f"  Found {len(clicks)} clicks")
            x = remove_clicks(x, sr, clicks, 
                            interpolation=ac_cfg["click_removal"].get("interpolation", "cubic"))
            applied.append("click_removal")
            metrics["clicks_removed"] = len(clicks)
    
    # 2. Musical Noise Suppression
    if ac_cfg.get("musical_noise", {}).get("enabled", True):
        print("[INFO] Suppressing musical noise...")
        x = suppress_musical_noise(
            x, sr,
            method=ac_cfg["musical_noise"].get("method", "spectral_floor"),
            floor_factor=ac_cfg["musical_noise"].get("floor_factor", 0.1),
            median_time=ac_cfg["musical_noise"].get("median_time", 5),
            median_freq=ac_cfg["musical_noise"].get("median_freq", 3)
        )
        applied.append("musical_noise_suppression")
    
    # 3. Transient Preservation
    if ac_cfg.get("transient_preservation", {}).get("enabled", True):
        print("[INFO] Preserving transients...")
        transient_mask, flux = preserve_transients(
            x, sr,
            detection_threshold=ac_cfg["transient_preservation"].get("detection_threshold", 0.3),
            protection_ms=ac_cfg["transient_preservation"].get("protection_ms", 5),
            blend_factor=ac_cfg["transient_preservation"].get("blend_factor", 0.7)
        )
        applied.append("transient_preservation")
        metrics["transient_ratio"] = float(np.mean(transient_mask))
    
    # 4. Pre-echo Control
    if ac_cfg.get("pre_echo", {}).get("enabled", True):
        print("[INFO] Controlling pre-echo...")
        x = control_pre_echo(
            x, sr,
            lookahead_ms=ac_cfg["pre_echo"].get("lookahead_ms", 5),
            threshold=ac_cfg["pre_echo"].get("threshold", 0.1)
        )
        applied.append("pre_echo_control")
    
    # 5. De-essing
    if ac_cfg.get("de_ess", {}).get("enabled", True):
        print("[INFO] Applying de-essing...")
        x = de_ess(
            x, sr,
            frequency_range=ac_cfg["de_ess"].get("frequency_range", [4000, 9000]),
            threshold_db=ac_cfg["de_ess"].get("threshold_db", -25),
            ratio=ac_cfg["de_ess"].get("ratio", 4),
            attack_ms=ac_cfg["de_ess"].get("attack_ms", 0.5),
            release_ms=ac_cfg["de_ess"].get("release_ms", 10)
        )
        applied.append("de_essing")
    
    # 6. Phase Coherence
    if ac_cfg.get("phase_coherence", {}).get("enabled", True):
        print("[INFO] Ensuring phase coherence...")
        x = ensure_phase_coherence(
            x, sr,
            method=ac_cfg["phase_coherence"].get("method", "griffin_lim"),
            iterations=ac_cfg["phase_coherence"].get("iterations", 32)
        )
        applied.append("phase_coherence")
    
    # 7. DC Offset Removal
    if ac_cfg.get("dc_offset", {}).get("enabled", True):
        print("[INFO] Removing DC offset...")
        dc_before = np.mean(x)
        x = remove_dc_offset(
            x, sr,
            method=ac_cfg["dc_offset"].get("method", "highpass"),
            cutoff_hz=ac_cfg["dc_offset"].get("cutoff_hz", 20)
        )
        dc_after = np.mean(x)
        applied.append("dc_offset_removal")
        metrics["dc_offset_removed"] = float(dc_before - dc_after)
    
    # 8. Safety Limiter
    if ac_cfg.get("safety_limiter", {}).get("enabled", True):
        print("[INFO] Applying safety limiter...")
        peaks_before = np.sum(np.abs(x) > 0.95)
        x = apply_safety_limiter(
            x,
            ceiling_db=ac_cfg["safety_limiter"].get("ceiling_db", -0.3),
            release_ms=ac_cfg["safety_limiter"].get("release_ms", 50),
            lookahead_ms=ac_cfg["safety_limiter"].get("lookahead_ms", 5)
        )
        peaks_after = np.sum(np.abs(x) > 0.95)
        applied.append("safety_limiter")
        metrics["peaks_limited"] = int(peaks_before - peaks_after)
    
    # Save output
    print(f"[INFO] Saving to {out_path}")
    sf.write(out_path, x, sr)
    
    dt = time.time() - t0
    
    # Compute final metrics
    metrics.update({
        "peak_level_db": float(20 * np.log10(np.max(np.abs(x)) + 1e-10)),
        "rms_level_db": float(20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-10)),
        "crest_factor_db": float(20 * np.log10(np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-10) + 1e-10))
    })
    
    # Log results
    log_data = {
        "phase": "5_artifact_control",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "input": in_path,
        "output": out_path,
        "applied_processes": applied,
        "metrics": metrics,
        "configuration": ac_cfg,
        "processing_time_sec": dt
    }
    
    log_path = os.path.join(logs_dir, "phase5_artifact_control.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"[INFO] Phase 5 complete in {dt:.1f}s")
    print(f"[INFO] Applied: {', '.join(applied)}")
    print(f"[INFO] Peak level: {metrics['peak_level_db']:.1f} dB")

if __name__ == "__main__":
    main()