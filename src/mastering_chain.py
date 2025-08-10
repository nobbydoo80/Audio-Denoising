#!/usr/bin/env python
"""
Phase 6: Mastering Chain
LUFS normalization, true peak limiting, EQ, multiband compression, and final polish
"""
import argparse
import os
import json
import time
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal

def set_seed(seed: int):
    """Set deterministic seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)

def ensure_dirs(paths):
    """Create directories if they don't exist"""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def compute_lufs(x, sr, integrated=True, short_term=False, momentary=False):
    """
    Compute LUFS (Loudness Units relative to Full Scale)
    Simplified implementation - for production use pyloudnorm
    """
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        
        results = {}
        if integrated:
            results['integrated'] = meter.integrated_loudness(x)
        # Note: pyloudnorm doesn't directly provide short-term/momentary
        # These would require windowed analysis
        
        return results
        
    except ImportError:
        # Fallback: simplified LUFS approximation using K-weighting
        print("[WARN] pyloudnorm not available, using simplified LUFS approximation")
        
        # K-weighting filter (simplified)
        # High shelf at 2kHz, +4dB
        # High-pass at 50Hz
        
        # High-pass filter
        sos_hp = signal.butter(2, 50, btype='high', fs=sr, output='sos')
        x_filtered = signal.sosfilt(sos_hp, x)
        
        # High shelf approximation (boost high frequencies)
        freq_shelf = 2000
        sos_shelf = signal.butter(1, freq_shelf, btype='high', fs=sr, output='sos')
        x_shelf = signal.sosfilt(sos_shelf, x_filtered) * 1.5  # ~4dB boost
        x_filtered = x_filtered + (x_shelf - x_filtered) * 0.5
        
        # Mean square
        mean_square = np.mean(x_filtered ** 2)
        
        # Convert to LUFS (approximate)
        if mean_square > 0:
            lufs = -0.691 + 10 * np.log10(mean_square)
        else:
            lufs = -70.0
        
        return {'integrated': lufs}

def normalize_lufs(x, sr, target_lufs=-16.0):
    """
    Normalize audio to target LUFS
    """
    current_lufs = compute_lufs(x, sr, integrated=True).get('integrated', -70)
    
    if current_lufs == -70:
        print("[WARN] Signal too quiet, skipping LUFS normalization")
        return x
    
    # Calculate gain needed
    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20)
    
    print(f"[INFO] Current LUFS: {current_lufs:.1f}, Target: {target_lufs:.1f}, Gain: {gain_db:.1f} dB")
    
    # Apply gain
    x_normalized = x * gain_linear
    
    # Prevent clipping
    if np.max(np.abs(x_normalized)) > 0.99:
        x_normalized = x_normalized * 0.99 / np.max(np.abs(x_normalized))
    
    return x_normalized

def true_peak_limit(x, sr, ceiling_dbtp=-1.0, release_ms=50, lookahead_ms=5):
    """
    True Peak limiting (considers inter-sample peaks)
    """
    ceiling_linear = 10 ** (ceiling_dbtp / 20)
    
    # Oversample to detect inter-sample peaks
    oversample_factor = 4
    x_oversampled = signal.resample(x, len(x) * oversample_factor)
    
    # Find peaks
    peaks = np.abs(x_oversampled)
    max_peak = np.max(peaks)
    
    if max_peak <= ceiling_linear:
        return x
    
    # Calculate required gain reduction
    gr = ceiling_linear / max_peak
    
    print(f"[INFO] True peak limiting: {20*np.log10(max_peak):.1f} dBTP -> {ceiling_dbtp:.1f} dBTP")
    
    # Apply limiting with lookahead
    lookahead_samples = int(lookahead_ms * sr / 1000)
    release_samples = int(release_ms * sr / 1000)
    
    # Simple lookahead limiter
    x_limited = x.copy()
    envelope = np.abs(x)
    
    # Smooth envelope with lookahead
    for i in range(len(envelope)):
        window_end = min(i + lookahead_samples, len(envelope))
        envelope[i] = np.max(envelope[i:window_end])
    
    # Apply gain reduction where needed
    gain_curve = np.ones_like(envelope)
    above_threshold = envelope > ceiling_linear
    
    if np.any(above_threshold):
        gain_curve[above_threshold] = ceiling_linear / envelope[above_threshold]
        
        # Smooth gain curve (release)
        for i in range(1, len(gain_curve)):
            if gain_curve[i] > gain_curve[i-1]:
                # Release
                alpha = np.exp(-1.0 / release_samples)
                gain_curve[i] = alpha * gain_curve[i-1] + (1 - alpha) * gain_curve[i]
    
    x_limited = x * gain_curve
    
    return x_limited

def apply_tilt_eq(x, sr, tilt_db=0):
    """
    Apply tilt EQ (positive = brighter, negative = darker)
    """
    if abs(tilt_db) < 0.1:
        return x
    
    # Create tilted frequency response
    # Low shelf at 100Hz, high shelf at 10kHz
    
    if tilt_db > 0:
        # Brighter: boost highs, cut lows
        sos_low = signal.butter(1, 100, btype='low', fs=sr, output='sos')
        sos_high = signal.butter(1, 10000, btype='high', fs=sr, output='sos')
        
        low_band = signal.sosfilt(sos_low, x)
        high_band = signal.sosfilt(sos_high, x)
        mid_band = x - low_band - high_band
        
        # Apply tilt
        gain_low = 10 ** (-tilt_db / 40)  # Reduce lows
        gain_high = 10 ** (tilt_db / 40)  # Boost highs
        
        x_tilted = low_band * gain_low + mid_band + high_band * gain_high
        
    else:
        # Darker: boost lows, cut highs
        sos_low = signal.butter(1, 100, btype='low', fs=sr, output='sos')
        sos_high = signal.butter(1, 10000, btype='high', fs=sr, output='sos')
        
        low_band = signal.sosfilt(sos_low, x)
        high_band = signal.sosfilt(sos_high, x)
        mid_band = x - low_band - high_band
        
        gain_low = 10 ** (-tilt_db / 40)  # Boost lows
        gain_high = 10 ** (tilt_db / 40)  # Reduce highs
        
        x_tilted = low_band * gain_low + mid_band + high_band * gain_high
    
    return x_tilted

def multiband_compress(x, sr, bands_config):
    """
    Multiband compression
    """
    compressed_bands = []
    
    for band_cfg in bands_config:
        freq_range = band_cfg['range']
        threshold_db = band_cfg['threshold_db']
        ratio = band_cfg['ratio']
        attack_ms = band_cfg['attack_ms']
        release_ms = band_cfg['release_ms']
        makeup_db = band_cfg['makeup_db']
        
        # Create band-pass filter
        if freq_range[0] == 20:
            # Low-pass for lowest band
            sos = signal.butter(4, freq_range[1], btype='low', fs=sr, output='sos')
        elif freq_range[1] == 20000:
            # High-pass for highest band
            sos = signal.butter(4, freq_range[0], btype='high', fs=sr, output='sos')
        else:
            # Band-pass for mid bands
            sos = signal.butter(4, freq_range, btype='band', fs=sr, output='sos')
        
        # Extract band
        band = signal.sosfilt(sos, x)
        
        # Compress band
        compressed = compress_audio(band, sr, threshold_db, ratio, attack_ms, release_ms, makeup_db)
        compressed_bands.append(compressed)
    
    # Sum compressed bands
    x_compressed = np.sum(compressed_bands, axis=0)
    
    # Prevent clipping
    if np.max(np.abs(x_compressed)) > 0.99:
        x_compressed = x_compressed * 0.99 / np.max(np.abs(x_compressed))
    
    return x_compressed

def compress_audio(x, sr, threshold_db=-20, ratio=2, attack_ms=5, release_ms=50, makeup_db=0):
    """
    Simple compressor
    """
    threshold_linear = 10 ** (threshold_db / 20)
    makeup_linear = 10 ** (makeup_db / 20)
    
    # Envelope follower
    envelope = np.abs(x)
    
    # Smooth envelope
    attack_samples = int(attack_ms * sr / 1000)
    release_samples = int(release_ms * sr / 1000)
    
    smoothed = np.zeros_like(envelope)
    for i in range(1, len(envelope)):
        if envelope[i] > smoothed[i-1]:
            # Attack
            alpha = 1.0 - np.exp(-1.0 / max(1, attack_samples))
        else:
            # Release
            alpha = 1.0 - np.exp(-1.0 / max(1, release_samples))
        smoothed[i] = alpha * envelope[i] + (1 - alpha) * smoothed[i-1]
    
    # Calculate gain reduction
    gain = np.ones_like(smoothed)
    above_threshold = smoothed > threshold_linear
    
    if np.any(above_threshold):
        # Compression ratio
        excess = smoothed[above_threshold] - threshold_linear
        compressed_excess = excess / ratio
        gain[above_threshold] = (threshold_linear + compressed_excess) / smoothed[above_threshold]
    
    # Apply compression and makeup gain
    x_compressed = x * gain * makeup_linear
    
    return x_compressed

def apply_dither(x, bit_depth=16, dither_type='triangular', noise_shaping=True):
    """
    Apply dither for bit depth reduction
    """
    # Calculate quantization step
    max_val = 2 ** (bit_depth - 1)
    
    if dither_type == 'none':
        # Simple quantization
        x_dithered = np.round(x * max_val) / max_val
        
    elif dither_type == 'rectangular':
        # Rectangular (uniform) dither
        dither = (np.random.random(len(x)) - 0.5) / max_val
        x_dithered = np.round((x + dither) * max_val) / max_val
        
    elif dither_type == 'triangular':
        # Triangular dither (sum of two rectangular)
        dither1 = (np.random.random(len(x)) - 0.5) / max_val
        dither2 = (np.random.random(len(x)) - 0.5) / max_val
        dither = (dither1 + dither2) / 2
        x_dithered = np.round((x + dither) * max_val) / max_val
        
    elif dither_type == 'shaped':
        # Noise-shaped dither (simplified)
        error = np.zeros_like(x)
        x_dithered = np.zeros_like(x)
        
        for i in range(len(x)):
            # Add shaped noise from previous error
            if i > 0:
                shaped_noise = error[i-1] * 0.5  # First-order shaping
            else:
                shaped_noise = 0
            
            # Add triangular dither
            dither = (np.random.random() - 0.5 + np.random.random() - 0.5) / (2 * max_val)
            
            # Quantize
            val = x[i] + shaped_noise + dither
            x_dithered[i] = np.round(val * max_val) / max_val
            
            # Store error for next sample
            error[i] = x[i] - x_dithered[i]
    else:
        x_dithered = x
    
    return x_dithered

def stereo_width_control(x_stereo, width=1.0, bass_mono_freq=120, sr=44100):
    """
    Control stereo width (if stereo input)
    width: 0=mono, 1=normal, >1=wider
    """
    if x_stereo.ndim != 2 or x_stereo.shape[1] != 2:
        return x_stereo
    
    # Mid-Side processing
    mid = (x_stereo[:, 0] + x_stereo[:, 1]) / 2
    side = (x_stereo[:, 0] - x_stereo[:, 1]) / 2
    
    # Apply width
    side = side * width
    
    # Bass mono (mono below specified frequency)
    if bass_mono_freq > 0:
        sos = signal.butter(2, bass_mono_freq, btype='low', fs=sr, output='sos')
        bass_mid = signal.sosfilt(sos, mid)
        bass_side = signal.sosfilt(sos, side)
        
        # Remove bass from side
        side = side - bass_side
    
    # Convert back to L-R
    left = mid + side
    right = mid - side
    
    return np.stack([left, right], axis=1)

def main():
    ap = argparse.ArgumentParser(description="Phase 6: Mastering Chain")
    ap.add_argument("--config", required=True, help="Path to config file")
    ap.add_argument("--input", default=None, help="Input audio")
    ap.add_argument("--output", default=None)
    ap.add_argument("--profile", default=None, help="Mastering profile (streaming/podcast/broadcast/club)")
    args = ap.parse_args()
    
    # Load config
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Check if phase is enabled
    if not cfg.get("phases", {}).get("phase6", {}).get("enabled", True):
        print("[INFO] Phase 6 (Mastering) is disabled in config, skipping")
        return
    
    # Set seed
    seed = cfg.get("mastering", {}).get("seed", cfg.get("global", {}).get("seed", 42))
    set_seed(seed)
    
    # Setup paths
    outputs_dir = cfg["paths"]["outputs_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    reports_dir = cfg["paths"]["reports_dir"]
    ensure_dirs([outputs_dir, logs_dir, reports_dir])
    
    # Determine input
    prefer = [
        os.path.join(outputs_dir, "phase5_artifact_controlled.wav"),
        os.path.join(outputs_dir, "phase4_ai_enhanced.wav"),
        os.path.join(outputs_dir, "phase3_hpss.wav"),
        os.path.join(outputs_dir, "phase2c_wiener.wav"),
        cfg["paths"]["input"]
    ]
    in_path = args.input or next((p for p in prefer if os.path.exists(p)), cfg["paths"]["input"])
    
    # Determine output based on profile
    master_cfg = cfg.get("mastering", {})
    
    if args.profile:
        profile_name = args.profile
    else:
        profile_name = "streaming"  # Default
    
    # Get profile settings
    profiles = master_cfg.get("profiles", {})
    if profile_name in profiles:
        profile = profiles[profile_name]
        target_lufs = profile.get("lufs", -16)
        true_peak = profile.get("true_peak", -1)
        output_format = profile.get("format", "wav")
    else:
        target_lufs = master_cfg.get("lufs_normalization", {}).get("target_lufs", -16)
        true_peak = master_cfg.get("true_peak_limiting", {}).get("ceiling_dbtp", -1)
        output_format = "wav"
    
    out_path = args.output or os.path.join(outputs_dir, f"final_mastered_{profile_name}.{output_format}")
    
    # Load audio
    print(f"[INFO] Loading {in_path}")
    x, sr = sf.read(in_path)
    original_shape = x.shape
    
    # Convert to mono for processing if needed
    if x.ndim > 1:
        x_mono = np.mean(x, axis=1)
        is_stereo = True
    else:
        x_mono = x
        is_stereo = False
    
    print(f"[INFO] Mastering with profile '{profile_name}': LUFS={target_lufs}, True Peak={true_peak} dBTP")
    t0 = time.time()
    
    # Track applied processes
    applied = []
    metrics_before = {}
    metrics_after = {}
    
    # Measure initial state
    lufs_before = compute_lufs(x_mono, sr, integrated=True).get('integrated', -70)
    peak_before = 20 * np.log10(np.max(np.abs(x_mono)) + 1e-10)
    metrics_before['lufs'] = lufs_before
    metrics_before['peak_db'] = peak_before
    
    # 1. EQ (Tilt)
    if master_cfg.get("eq", {}).get("enabled", True):
        print("[INFO] Applying EQ...")
        tilt_db = master_cfg["eq"].get("tilt_db", 0)
        x_mono = apply_tilt_eq(x_mono, sr, tilt_db)
        applied.append(f"tilt_eq_{tilt_db:+.1f}dB")
    
    # 2. Multiband Compression
    if master_cfg.get("multiband_compression", {}).get("enabled", True):
        print("[INFO] Applying multiband compression...")
        bands = master_cfg["multiband_compression"].get("bands", [
            {"range": [20, 250], "threshold_db": -25, "ratio": 3, 
             "attack_ms": 10, "release_ms": 100, "makeup_db": 0},
            {"range": [250, 1000], "threshold_db": -20, "ratio": 2,
             "attack_ms": 5, "release_ms": 50, "makeup_db": 0},
            {"range": [1000, 4000], "threshold_db": -20, "ratio": 2,
             "attack_ms": 3, "release_ms": 30, "makeup_db": 0},
            {"range": [4000, 20000], "threshold_db": -25, "ratio": 3,
             "attack_ms": 1, "release_ms": 20, "makeup_db": 0}
        ])
        x_mono = multiband_compress(x_mono, sr, bands)
        applied.append("multiband_compression")
    
    # 3. LUFS Normalization
    if master_cfg.get("lufs_normalization", {}).get("enabled", True):
        print("[INFO] Normalizing to target LUFS...")
        x_mono = normalize_lufs(x_mono, sr, target_lufs)
        applied.append(f"lufs_norm_{target_lufs}")
    
    # 4. True Peak Limiting
    if master_cfg.get("true_peak_limiting", {}).get("enabled", True):
        print("[INFO] Applying true peak limiting...")
        x_mono = true_peak_limit(
            x_mono, sr,
            ceiling_dbtp=true_peak,
            release_ms=master_cfg["true_peak_limiting"].get("release_ms", 50),
            lookahead_ms=master_cfg["true_peak_limiting"].get("lookahead_ms", 5)
        )
        applied.append(f"true_peak_{true_peak}dBTP")
    
    # 5. Dithering (if reducing bit depth)
    target_bit_depth = cfg.get("audio", {}).get("target_bit_depth", 16)
    if master_cfg.get("dithering", {}).get("enabled", True) and target_bit_depth < 24:
        print(f"[INFO] Applying dither for {target_bit_depth}-bit output...")
        x_mono = apply_dither(
            x_mono,
            bit_depth=target_bit_depth,
            dither_type=master_cfg["dithering"].get("type", "triangular"),
            noise_shaping=master_cfg["dithering"].get("noise_shaping", True)
        )
        applied.append(f"dither_{target_bit_depth}bit")
    
    # Convert back to stereo if needed
    if is_stereo and original_shape[1] == 2:
        # Simple mono to stereo
        x_final = np.stack([x_mono, x_mono], axis=1)
        
        # Apply stereo width control if configured
        if master_cfg.get("stereo_width", {}).get("enabled", False):
            width = master_cfg["stereo_width"].get("width", 1.0)
            bass_mono = master_cfg["stereo_width"].get("bass_mono_freq", 120)
            x_final = stereo_width_control(x_final, width, bass_mono, sr)
            applied.append(f"stereo_width_{width}")
    else:
        x_final = x_mono
    
    # Measure final state
    lufs_after = compute_lufs(x_mono, sr, integrated=True).get('integrated', -70)
    peak_after = 20 * np.log10(np.max(np.abs(x_mono)) + 1e-10)
    metrics_after['lufs'] = lufs_after
    metrics_after['peak_db'] = peak_after
    metrics_after['true_peak_dbtp'] = peak_after  # Simplified, should use oversampling
    
    # Save output
    print(f"[INFO] Saving mastered audio to {out_path}")
    
    # Determine subtype based on bit depth
    if output_format == 'wav':
        if target_bit_depth == 16:
            subtype = 'PCM_16'
        elif target_bit_depth == 24:
            subtype = 'PCM_24'
        else:
            subtype = 'PCM_32'
        sf.write(out_path, x_final, sr, subtype=subtype)
    else:
        sf.write(out_path, x_final, sr)
    
    # Also save the standard final output path
    final_path = cfg["paths"].get("final_output", os.path.join(outputs_dir, "final_mastered.wav"))
    if final_path != out_path:
        import shutil
        shutil.copy2(out_path, final_path)
        print(f"[INFO] Also saved to {final_path}")
    
    dt = time.time() - t0
    
    # Compliance check
    compliance = {
        "lufs_target_met": abs(lufs_after - target_lufs) < 0.5,
        "true_peak_met": peak_after <= true_peak + 0.1,
        "ebu_r128_compliant": target_lufs == -23 and true_peak == -1,
        "streaming_ready": target_lufs >= -16 and target_lufs <= -14 and true_peak <= -1
    }
    
    # Log results
    log_data = {
        "phase": "6_mastering",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "input": in_path,
        "output": out_path,
        "profile": profile_name,
        "applied_processes": applied,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "compliance": compliance,
        "configuration": {
            "target_lufs": target_lufs,
            "true_peak_dbtp": true_peak,
            "bit_depth": target_bit_depth,
            "format": output_format
        },
        "processing_time_sec": dt
    }
    
    log_path = os.path.join(logs_dir, "phase6_mastering.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"[INFO] Phase 6 complete in {dt:.1f}s")
    print(f"[INFO] Final LUFS: {lufs_after:.1f} (target: {target_lufs})")
    print(f"[INFO] Final Peak: {peak_after:.1f} dB (limit: {true_peak} dBTP)")
    print(f"[INFO] Compliance: Streaming={compliance['streaming_ready']}, EBU={compliance['ebu_r128_compliant']}")

if __name__ == "__main__":
    main()