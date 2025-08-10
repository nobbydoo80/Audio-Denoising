#!/usr/bin/env python
import argparse, os, json
from pathlib import Path
import numpy as np
import soundfile as sf

def ensure_dirs(paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def stft_mag(x, sr, n_fft, hop_length, win_length, window="hann", center=True):
    from scipy.signal import stft, get_window
    f, t, Z = stft(x, fs=sr, nperseg=win_length, noverlap=win_length-hop_length, window=get_window(window, win_length), padded=center, return_onesided=True, boundary=None)
    return f, t, np.abs(Z)

def smooth_2d(M, tf=3, ff=5):
    # simple box filter smoothing
    from scipy.ndimage import uniform_filter
    return uniform_filter(M, size=(ff, tf), mode="nearest")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--noise", default=None)
    ap.add_argument("--output", default=None, help="Optional path to save learned spectral noise profile (npz)")
    args = ap.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    noise_path = args.noise or cfg["paths"]["noise"]
    outputs_dir = cfg["paths"]["outputs_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    reports_dir = cfg["paths"]["reports_dir"]
    ensure_dirs([outputs_dir, logs_dir, reports_dir])

    # Load noise sample
    n, sr = sf.read(noise_path)
    if n.ndim > 1: n = np.mean(n, axis=1)

    # STFT config (reuse analysis config)
    st = cfg["analysis"]["stft"]
    f, t, N = stft_mag(n, sr, st["n_fft"], st["hop_length"], st["win_length"], st["window"], st["center"])

    # Percentile-based spectral estimate with smoothing and spectral floor
    perc = cfg["noise_profile"]["percentile_level"]
    floor = cfg["noise_profile"]["spectral_floor"]
    tf = cfg["noise_profile"]["smoothing"]["time_frames"]
    ff = cfg["noise_profile"]["smoothing"]["freq_bins"]

    per_frame_noise = np.percentile(N, perc, axis=0)  # time-varying baseline per frame
    global_noise = np.percentile(N, perc, axis=1)     # per-frequency baseline

    # Form a 2D baseline by outer sum approximation and then smooth
    baseline = np.outer(global_noise, np.ones_like(per_frame_noise))
    baseline = np.maximum(baseline, floor * np.max(N))
    baseline = smooth_2d(baseline, tf=tf, ff=ff)

    profile = {
        "samplerate": sr,
        "frequencies": f,
        "times": t,
        "baseline_mag": baseline.astype(np.float32),
        "percentile": perc,
        "spectral_floor": floor
    }

    out_npz = args.output or os.path.join(outputs_dir, "phase1_noise_profile.npz")
    np.savez_compressed(out_npz, **profile)

    with open(os.path.join(logs_dir, "phase1_noise_profile.json"), "w") as fjson:
        json.dump({
            "noise_input": noise_path,
            "output_npz": out_npz,
            "percentile_level": perc,
            "spectral_floor": floor,
            "smoothing": {"time_frames": tf, "freq_bins": ff}
        }, fjson, indent=2)

    print(f"Phase 1 noise profile learned -> {out_npz}")

if __name__ == "__main__":
    main()