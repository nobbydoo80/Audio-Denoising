#!/usr/bin/env python
"""
Phase 3: Harmonic-Percussive Source Separation (HPSS)
Separates harmonic (tonal/speech) and percussive (transient/noise) components
"""
import argparse
import os
import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import medfilt2d
from scipy.ndimage import median_filter

def set_seed(seed: int):
    """Set deterministic seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)

def ensure_dirs(paths):
    """Create directories if they don't exist"""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def stft_transform(x, sr, n_fft=2048, hop_length=512):
    """Compute STFT"""
    from scipy.signal import stft
    f, t, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length, 
                   window='hann', boundary=None, padded=True, return_onesided=True)
    return f, t, Z

def istft_transform(Z, sr, n_fft=2048, hop_length=512):
    """Compute inverse STFT"""
    from scipy.signal import istft
    _, y = istft(Z, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length,
                 window='hann', input_onesided=True)
    return y

def median_filtering_hpss(S, margin_h=1.0, margin_p=5.0, kernel_size=31, power=2.0):
    """
    Median-filtering HPSS (fallback when librosa not available)
    
    Parameters:
    - S: Magnitude spectrogram
    - margin_h: Harmonic margin factor
    - margin_p: Percussive margin factor  
    - kernel_size: Median filter kernel size
    - power: Exponent for Wiener filter
    """
    # Apply median filters
    # Harmonic: horizontal median filter (enhance horizontal lines)
    S_harmonic = median_filter(S, size=(1, kernel_size), mode='reflect')
    
    # Percussive: vertical median filter (enhance vertical lines)
    S_percussive = median_filter(S, size=(kernel_size, 1), mode='reflect')
    
    # Soft masks using margins
    mask_h = (S_harmonic * margin_h) ** power
    mask_p = (S_percussive * margin_p) ** power
    total = mask_h + mask_p + 1e-10
    
    mask_h = mask_h / total
    mask_p = mask_p / total
    
    return mask_h, mask_p

def librosa_hpss(S, margin=(1.0, 5.0), kernel_size=31, power=2.0, mask=True):
    """
    Librosa HPSS implementation (if available)
    """
    try:
        import librosa
        # Use librosa's implementation
        S_h, S_p = librosa.decompose.hpss(S, margin=margin, kernel_size=kernel_size,
                                          power=power, mask=mask)
        if mask:
            # Return masks
            total = S_h + S_p + 1e-10
            return S_h / total, S_p / total
        else:
            return S_h, S_p
    except ImportError:
        print("[WARN] librosa not available, using fallback median filtering")
        return median_filtering_hpss(S, margin[0], margin[1], kernel_size, power)

def multiband_hpss(S, f, bands, margin_h=1.0, margin_p=5.0, kernel_size=31):
    """
    Apply HPSS separately to different frequency bands
    """
    mask_h = np.zeros_like(S)
    mask_p = np.zeros_like(S)
    
    for band_low, band_high in bands:
        # Find frequency indices for this band
        band_mask = (f >= band_low) & (f <= band_high)
        if not np.any(band_mask):
            continue
            
        # Extract band
        S_band = S[band_mask, :]
        
        # Apply HPSS to this band
        mask_h_band, mask_p_band = median_filtering_hpss(
            S_band, margin_h, margin_p, kernel_size
        )
        
        # Store results
        mask_h[band_mask, :] = mask_h_band
        mask_p[band_mask, :] = mask_p_band
    
    return mask_h, mask_p

def apply_masks(Z, mask_h, mask_p, mask_type='soft'):
    """
    Apply separation masks to complex spectrogram
    
    Parameters:
    - Z: Complex STFT
    - mask_h: Harmonic mask
    - mask_p: Percussive mask
    - mask_type: 'soft', 'hard', or 'binary'
    """
    S = np.abs(Z)
    
    if mask_type == 'binary':
        # Binary masks
        mask_h = (mask_h > 0.5).astype(float)
        mask_p = (mask_p > 0.5).astype(float)
    elif mask_type == 'hard':
        # Winner-take-all
        mask_h = (mask_h > mask_p).astype(float)
        mask_p = 1.0 - mask_h
    # else: soft masks as-is
    
    Z_h = Z * mask_h
    Z_p = Z * mask_p
    
    return Z_h, Z_p

def preserve_transients(y_harmonic, y_percussive, y_original, threshold=0.3, blend=0.7):
    """
    Preserve transients from original signal
    """
    # Ensure all signals have same length first
    min_len = min(len(y_harmonic), len(y_percussive), len(y_original))
    y_harmonic = y_harmonic[:min_len]
    y_percussive = y_percussive[:min_len]
    y_original = y_original[:min_len]
    
    # Simple transient detection using energy difference
    window_size = 2048
    hop = 512
    
    # Compute local energy
    def local_energy(signal):
        energy = []
        for i in range(0, len(signal) - window_size, hop):
            frame = signal[i:i+window_size]
            energy.append(np.sqrt(np.mean(frame**2)))
        return np.array(energy)
    
    energy_orig = local_energy(y_original)
    energy_harm = local_energy(y_harmonic)
    
    # Ensure same length
    min_len = min(len(energy_orig), len(energy_harm))
    energy_orig = energy_orig[:min_len]
    energy_harm = energy_harm[:min_len]
    
    # Detect transients where original has more energy than harmonic
    transient_frames = (energy_orig - energy_harm) > threshold * np.max(energy_orig)
    
    # Blend in transients
    y_result = y_harmonic.copy()
    for i, is_transient in enumerate(transient_frames):
        if is_transient:
            start = i * hop
            end = min(start + window_size, len(y_result), len(y_original))
            # Ensure we don't exceed array bounds
            actual_len = end - start
            # Blend original with harmonic
            y_result[start:end] = (blend * y_original[start:end] + 
                                   (1 - blend) * y_harmonic[start:end])
    
    return y_result

def main():
    ap = argparse.ArgumentParser(description="Phase 3: HPSS Separation")
    ap.add_argument("--config", required=True, help="Path to config file")
    ap.add_argument("--input", default=None, help="Input audio (defaults to Phase 2 output)")
    ap.add_argument("--output_harmonic", default=None)
    ap.add_argument("--output_percussive", default=None)
    ap.add_argument("--output_combined", default=None)
    args = ap.parse_args()
    
    # Load config
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Check if phase is enabled
    if not cfg.get("phases", {}).get("phase3", {}).get("enabled", True):
        print("[INFO] Phase 3 (HPSS) is disabled in config, skipping")
        return
    
    # Set seed for reproducibility
    seed = cfg.get("hpss", {}).get("seed", cfg.get("global", {}).get("seed", 42))
    set_seed(seed)
    
    # Setup paths
    outputs_dir = cfg["paths"]["outputs_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dirs([outputs_dir, logs_dir])
    
    # Determine input (prefer Phase 2c output)
    prefer = [
        os.path.join(outputs_dir, "phase2c_wiener.wav"),
        os.path.join(outputs_dir, "phase2b_spectral_subtract.wav"),
        os.path.join(outputs_dir, "phase2_wavelet.wav"),
        cfg["paths"]["input"]
    ]
    in_path = args.input or next((p for p in prefer if os.path.exists(p)), cfg["paths"]["input"])
    
    # Output paths
    out_harmonic = args.output_harmonic or os.path.join(outputs_dir, "phase3_harmonic.wav")
    out_percussive = args.output_percussive or os.path.join(outputs_dir, "phase3_percussive.wav")
    out_combined = args.output_combined or os.path.join(outputs_dir, "phase3_hpss.wav")
    
    # Load audio
    print(f"[INFO] Loading input from {in_path}")
    x, sr = sf.read(in_path)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    
    # Get HPSS parameters
    hpss_cfg = cfg.get("hpss", {})
    method = hpss_cfg.get("method", "median")
    margin_h = hpss_cfg.get("margin_harmonic", 1.0)
    margin_p = hpss_cfg.get("margin_percussive", 5.0)
    kernel_size = hpss_cfg.get("kernel_size", 31)
    power = hpss_cfg.get("power", 2.0)
    mask_type = hpss_cfg.get("mask_type", "soft")
    multiband_enabled = hpss_cfg.get("multiband", {}).get("enabled", False)
    preserve_trans = hpss_cfg.get("preserve_transients", True)
    reversible = hpss_cfg.get("reversible", True)
    
    # Start processing
    t0 = time.time()
    
    # STFT
    print("[INFO] Computing STFT...")
    f, t, Z = stft_transform(x, sr)
    S = np.abs(Z)
    
    # Apply HPSS
    print(f"[INFO] Applying HPSS (method={method})...")
    
    if multiband_enabled:
        bands = hpss_cfg.get("multiband", {}).get("bands", 
                            [[0, 250], [250, 2000], [2000, 8000], [8000, 22050]])
        mask_h, mask_p = multiband_hpss(S, f, bands, margin_h, margin_p, kernel_size)
    elif method == "librosa":
        mask_h, mask_p = librosa_hpss(S, (margin_h, margin_p), kernel_size, power, mask=True)
    else:  # median
        mask_h, mask_p = median_filtering_hpss(S, margin_h, margin_p, kernel_size, power)
    
    # Apply masks
    Z_h, Z_p = apply_masks(Z, mask_h, mask_p, mask_type)
    
    # Inverse STFT
    print("[INFO] Reconstructing audio...")
    y_harmonic = istft_transform(Z_h, sr)
    y_percussive = istft_transform(Z_p, sr)
    
    # Preserve transients if enabled
    if preserve_trans:
        print("[INFO] Preserving transients...")
        y_harmonic = preserve_transients(y_harmonic, y_percussive, x, 
                                        threshold=0.3, blend=0.7)
    
    # Normalize to prevent clipping
    max_val = max(np.max(np.abs(y_harmonic)), np.max(np.abs(y_percussive)))
    if max_val > 0.95:
        scale = 0.95 / max_val
        y_harmonic *= scale
        y_percussive *= scale
    
    # Save outputs
    print(f"[INFO] Saving outputs...")
    
    if reversible:
        # Save both components
        sf.write(out_harmonic, y_harmonic, sr)
        sf.write(out_percussive, y_percussive, sr)
        print(f"  - Harmonic: {out_harmonic}")
        print(f"  - Percussive: {out_percussive}")
    
    # Save combined (using harmonic as the denoised version)
    sf.write(out_combined, y_harmonic, sr)
    print(f"  - Combined: {out_combined}")
    
    # Compute metrics
    dt = time.time() - t0
    
    # Energy distribution
    energy_harmonic = np.sqrt(np.mean(y_harmonic**2))
    energy_percussive = np.sqrt(np.mean(y_percussive**2))
    energy_total = np.sqrt(np.mean(x**2))
    
    # Log results
    log_data = {
        "phase": "3_hpss",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "input": in_path,
        "outputs": {
            "harmonic": out_harmonic if reversible else None,
            "percussive": out_percussive if reversible else None,
            "combined": out_combined
        },
        "parameters": {
            "method": method,
            "margin_harmonic": margin_h,
            "margin_percussive": margin_p,
            "kernel_size": kernel_size,
            "power": power,
            "mask_type": mask_type,
            "multiband": multiband_enabled,
            "preserve_transients": preserve_trans
        },
        "metrics": {
            "processing_time_sec": dt,
            "energy_distribution": {
                "harmonic_rms": float(energy_harmonic),
                "percussive_rms": float(energy_percussive),
                "total_rms": float(energy_total),
                "harmonic_ratio": float(energy_harmonic / (energy_total + 1e-10))
            }
        }
    }
    
    log_path = os.path.join(logs_dir, "phase3_hpss.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"[INFO] Phase 3 complete in {dt:.1f}s")
    print(f"[INFO] Energy distribution: H={energy_harmonic:.3f}, P={energy_percussive:.3f}")

if __name__ == "__main__":
    main()