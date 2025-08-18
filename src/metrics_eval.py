#!/usr/bin/env python
"""
Metrics Evaluation Module
Computes SNR, PESQ, STOI, SI-SDR and other quality metrics
"""
import argparse
import os
import json
import time
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

def compute_snr(clean, noisy, method='global', segment_length_ms=30, sr=44100):
    """
    Compute Signal-to-Noise Ratio
    """
    if method == 'global':
        # Global SNR
        signal_power = np.mean(clean ** 2)
        noise = noisy - clean
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
        
        return snr
        
    elif method == 'segmental':
        # Segmental SNR
        segment_samples = int(segment_length_ms * sr / 1000)
        n_segments = len(clean) // segment_samples
        
        if n_segments == 0:
            return compute_snr(clean, noisy, 'global')
        
        snr_segments = []
        for i in range(n_segments):
            start = i * segment_samples
            end = start + segment_samples
            
            seg_clean = clean[start:end]
            seg_noisy = noisy[start:end]
            
            signal_power = np.mean(seg_clean ** 2)
            noise = seg_noisy - seg_clean
            noise_power = np.mean(noise ** 2)
            
            if noise_power > 0 and signal_power > 0:
                seg_snr = 10 * np.log10(signal_power / noise_power)
                snr_segments.append(seg_snr)
        
        if snr_segments:
            return np.mean(snr_segments)
        else:
            return 0.0

def compute_pesq(reference, degraded, sr, mode='wb'):
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality)
    Requires pesq package
    """
    try:
        from pesq import pesq as pesq_score
        
        # PESQ requires specific sample rates
        if mode == 'nb':  # Narrowband
            target_sr = 8000
        else:  # Wideband
            target_sr = 16000
        
        # Resample if needed
        if sr != target_sr:
            from scipy.signal import resample
            ratio = target_sr / sr
            reference = resample(reference, int(len(reference) * ratio))
            degraded = resample(degraded, int(len(degraded) * ratio))
            sr = target_sr
        
        # Ensure same length
        min_len = min(len(reference), len(degraded))
        reference = reference[:min_len]
        degraded = degraded[:min_len]
        
        # Compute PESQ
        score = pesq_score(sr, reference, degraded, mode)
        
        return score
        
    except ImportError:
        print("[WARN] pesq not available. Install with: pip install pesq")
        return None
    except Exception as e:
        print(f"[WARN] PESQ computation failed: {e}")
        return None

def compute_stoi(reference, degraded, sr, extended=True):
    """
    Compute STOI (Short-Time Objective Intelligibility)
    Requires pystoi package
    """
    try:
        from pystoi import stoi
        
        # STOI typically uses 10kHz sample rate
        target_sr = 10000
        
        # Resample if needed
        if sr != target_sr:
            from scipy.signal import resample
            ratio = target_sr / sr
            reference = resample(reference, int(len(reference) * ratio))
            degraded = resample(degraded, int(len(degraded) * ratio))
            sr = target_sr
        
        # Ensure same length
        min_len = min(len(reference), len(degraded))
        reference = reference[:min_len]
        degraded = degraded[:min_len]
        
        # Compute STOI
        score = stoi(reference, degraded, sr, extended=extended)
        
        return score
        
    except ImportError:
        print("[WARN] pystoi not available. Install with: pip install pystoi")
        return None
    except Exception as e:
        print(f"[WARN] STOI computation failed: {e}")
        return None

def compute_sisdr(reference, estimate):
    """
    Compute SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
    """
    # Ensure same length
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]
    
    # Remove mean
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    
    # Compute scaling factor
    alpha = np.dot(reference, estimate) / (np.dot(reference, reference) + 1e-10)
    
    # Scale reference
    target = alpha * reference
    
    # Compute SI-SDR
    target_energy = np.sum(target ** 2)
    error_energy = np.sum((estimate - target) ** 2)
    
    if error_energy > 0:
        sisdr = 10 * np.log10(target_energy / error_energy)
    else:
        sisdr = float('inf')
    
    return sisdr

def compute_lsd(reference, estimate, sr, n_fft=2048, hop_length=512):
    """
    Compute Log-Spectral Distance
    """
    # Compute spectrograms
    _, _, Z_ref = signal.stft(reference, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length)
    _, _, Z_est = signal.stft(estimate, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length)
    
    # Magnitude spectrograms
    S_ref = np.abs(Z_ref)
    S_est = np.abs(Z_est)
    
    # Log spectrograms (with floor to avoid log(0))
    log_S_ref = np.log10(S_ref + 1e-10)
    log_S_est = np.log10(S_est + 1e-10)
    
    # Squared difference
    squared_diff = (log_S_ref - log_S_est) ** 2
    
    # Mean over frequency, then RMS over time
    lsd_per_frame = np.sqrt(np.mean(squared_diff, axis=0))
    lsd = np.mean(lsd_per_frame)
    
    return lsd * 10  # Convert to dB

def align_signals(reference, estimate):
    """
    Align two signals using cross-correlation
    """
    # Compute cross-correlation
    correlation = signal.correlate(estimate, reference, mode='full')
    
    # Find peak
    lag = np.argmax(np.abs(correlation)) - len(reference) + 1
    
    # Align
    if lag > 0:
        # Estimate is delayed
        aligned_estimate = estimate[lag:]
        aligned_reference = reference[:len(aligned_estimate)]
    elif lag < 0:
        # Estimate is advanced
        aligned_reference = reference[-lag:]
        aligned_estimate = estimate[:len(aligned_reference)]
    else:
        aligned_reference = reference
        aligned_estimate = estimate
    
    # Ensure same length
    min_len = min(len(aligned_reference), len(aligned_estimate))
    aligned_reference = aligned_reference[:min_len]
    aligned_estimate = aligned_estimate[:min_len]
    
    return aligned_reference, aligned_estimate

def check_regression(current_metrics, baseline_metrics, tolerance_pct=5):
    """
    Check if current metrics show regression compared to baseline
    """
    regressions = []
    
    for metric_name in ['snr', 'pesq', 'stoi', 'sisdr']:
        if metric_name in current_metrics and metric_name in baseline_metrics:
            current = current_metrics[metric_name]
            baseline = baseline_metrics[metric_name]
            
            if current is not None and baseline is not None:
                # Check if regression exceeds tolerance
                if baseline > 0:
                    pct_change = ((current - baseline) / baseline) * 100
                    if pct_change < -tolerance_pct:
                        regressions.append({
                            'metric': metric_name,
                            'baseline': baseline,
                            'current': current,
                            'change_pct': pct_change
                        })
    
    return regressions

def main():
    ap = argparse.ArgumentParser(description="Metrics Evaluation")
    ap.add_argument("--config", required=True, help="Path to config file")
    ap.add_argument("--processed", default=None, help="Processed audio file")
    ap.add_argument("--reference", default=None, help="Reference audio file")
    ap.add_argument("--original", default=None, help="Original noisy audio")
    ap.add_argument("--output", default=None, help="Output metrics JSON path")
    args = ap.parse_args()
    
    # Load config
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Check if metrics are enabled
    if not cfg.get("metrics", {}).get("enabled", True):
        print("[INFO] Metrics evaluation is disabled in config, skipping")
        return
    
    # Set seed
    seed = cfg.get("metrics", {}).get("seed", cfg.get("global", {}).get("seed", 42))
    set_seed(seed)
    
    # Setup paths
    outputs_dir = cfg["paths"]["outputs_dir"]
    reports_dir = cfg["paths"]["reports_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dirs([reports_dir, logs_dir])
    
    # Determine files
    processed_path = args.processed or cfg["paths"].get("final_output", 
                                                        os.path.join(outputs_dir, "final_mastered.wav"))
    
    # Reference can be provided or we use the original input
    reference_path = args.reference or cfg["paths"].get("reference")
    original_path = args.original or cfg["paths"]["input"]
    
    if not os.path.exists(processed_path):
        print(f"[ERROR] Processed file not found: {processed_path}")
        return
    
    # Load processed audio
    print(f"[INFO] Loading processed audio: {processed_path}")
    processed, sr_p = sf.read(processed_path)
    if processed.ndim > 1:
        processed = np.mean(processed, axis=1)
    
    # Get metrics config
    metrics_cfg = cfg.get("metrics", {})
    require_reference = metrics_cfg.get("require_reference", False)
    
    # Initialize results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "processed_file": processed_path,
        "sample_rate": sr_p,
        "duration_sec": len(processed) / sr_p
    }
    
    t0 = time.time()
    
    # If we have a reference (clean) signal
    if reference_path and os.path.exists(reference_path):
        print(f"[INFO] Loading reference audio: {reference_path}")
        reference, sr_r = sf.read(reference_path)
        if reference.ndim > 1:
            reference = np.mean(reference, axis=1)
        
        # Resample if needed
        if sr_r != sr_p:
            print(f"[INFO] Resampling reference from {sr_r} to {sr_p} Hz")
            ratio = sr_p / sr_r
            reference = signal.resample(reference, int(len(reference) * ratio))
        
        # Align signals
        print("[INFO] Aligning signals...")
        reference, processed_aligned = align_signals(reference, processed)
        
        results["reference_file"] = reference_path
        
        # Compute reference-based metrics
        print("[INFO] Computing reference-based metrics...")
        
        # PESQ
        if metrics_cfg.get("pesq", {}).get("enabled", True):
            pesq_mode = metrics_cfg["pesq"].get("mode", "wb")
            pesq_score = compute_pesq(reference, processed_aligned, sr_p, pesq_mode)
            results["pesq"] = pesq_score
            print(f"  PESQ ({pesq_mode}): {pesq_score:.2f}" if pesq_score else "  PESQ: N/A")
        
        # STOI
        if metrics_cfg.get("stoi", {}).get("enabled", True):
            extended = metrics_cfg["stoi"].get("extended", True)
            stoi_score = compute_stoi(reference, processed_aligned, sr_p, extended)
            results["stoi"] = stoi_score
            print(f"  STOI: {stoi_score:.3f}" if stoi_score else "  STOI: N/A")
        
        # SI-SDR
        if metrics_cfg.get("sisdr", {}).get("enabled", True):
            sisdr = compute_sisdr(reference, processed_aligned)
            results["sisdr"] = sisdr
            print(f"  SI-SDR: {sisdr:.1f} dB")
        
        # LSD
        if metrics_cfg.get("spectral_distance", {}).get("enabled", True):
            lsd = compute_lsd(reference, processed_aligned, sr_p)
            results["lsd"] = lsd
            print(f"  LSD: {lsd:.2f} dB")
    
    elif require_reference:
        print("[WARN] Reference required but not provided, skipping reference-based metrics")
    
    # If we have the original noisy signal, compute SNR improvement
    if original_path and os.path.exists(original_path):
        print(f"[INFO] Loading original audio: {original_path}")
        original, sr_o = sf.read(original_path)
        if original.ndim > 1:
            original = np.mean(original, axis=1)
        
        # Resample if needed
        if sr_o != sr_p:
            ratio = sr_p / sr_o
            original = signal.resample(original, int(len(original) * ratio))
        
        results["original_file"] = original_path
        
        # SNR (if we have reference)
        if reference_path and os.path.exists(reference_path):
            if metrics_cfg.get("snr", {}).get("enabled", True):
                method = metrics_cfg["snr"].get("method", "segmental")
                segment_ms = metrics_cfg["snr"].get("segment_length_ms", 30)
                
                # Original SNR
                snr_original = compute_snr(reference, original, method, segment_ms, sr_p)
                
                # Processed SNR
                snr_processed = compute_snr(reference, processed_aligned, method, segment_ms, sr_p)
                
                # SNR improvement
                snr_improvement = snr_processed - snr_original
                
                results["snr_original"] = snr_original
                results["snr_processed"] = snr_processed
                results["snr_improvement"] = snr_improvement
                
                print(f"  SNR: {snr_original:.1f} -> {snr_processed:.1f} dB (improvement: {snr_improvement:.1f} dB)")
    
    # Check thresholds
    thresholds = metrics_cfg.get("thresholds", {})
    pass_fail = {}
    
    if "snr_improvement" in results:
        target = thresholds.get("snr_improvement_db", 15)
        pass_fail["snr_improvement"] = results["snr_improvement"] >= target
    
    if "pesq" in results and results["pesq"] is not None:
        target = thresholds.get("pesq_min", 3.5)
        pass_fail["pesq"] = results["pesq"] >= target
    
    if "stoi" in results and results["stoi"] is not None:
        target = thresholds.get("stoi_min", 0.85)
        pass_fail["stoi"] = results["stoi"] >= target
    
    if "sisdr" in results:
        target = thresholds.get("sisdr_min", 10)
        pass_fail["sisdr"] = results["sisdr"] >= target
    
    results["pass_fail"] = pass_fail
    results["overall_pass"] = all(pass_fail.values()) if pass_fail else None
    
    # Check for regression if baseline provided
    if metrics_cfg.get("regression", {}).get("enabled", True):
        baseline_path = metrics_cfg["regression"].get("baseline_path")
        if baseline_path and os.path.exists(baseline_path):
            with open(baseline_path, "r") as f:
                baseline = json.load(f)
            
            tolerance = metrics_cfg["regression"].get("tolerance_pct", 5)
            regressions = check_regression(results, baseline, tolerance)
            
            if regressions:
                results["regressions"] = regressions
                print(f"[WARN] Regressions detected: {len(regressions)} metrics")
            else:
                results["regressions"] = []
                print("[INFO] No regressions detected")
    
    dt = time.time() - t0
    results["processing_time_sec"] = dt
    
    # Save results
    output_path = args.output or cfg["paths"].get("metrics_path", 
                                                  os.path.join(reports_dir, "metrics.json"))
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Metrics evaluation complete in {dt:.1f}s")
    print(f"[INFO] Results saved to {output_path}")
    
    if results.get("overall_pass") is not None:
        print(f"[INFO] Overall: {'PASS' if results['overall_pass'] else 'FAIL'}")
    
    # Also log to phase log
    log_path = os.path.join(logs_dir, "metrics_evaluation.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()