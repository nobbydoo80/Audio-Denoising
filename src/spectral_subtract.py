#!/usr/bin/env python
import argparse, os, json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, get_window

def ensure_dirs(paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def load_noise_profile(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {
        "frequencies": d["frequencies"],
        "baseline_mag": d["baseline_mag"],
        "samplerate": int(d["samplerate"]),
        "percentile": float(d["percentile"]),
        "spectral_floor": float(d["spectral_floor"]),
    }

def spectral_flooring(mag, floor_frac):
    return np.maximum(mag, floor_frac * np.max(mag))

def smooth_2d(M, tf=3, ff=5):
    from scipy.ndimage import uniform_filter
    return uniform_filter(M, size=(ff, tf), mode="nearest")

def spectral_subtract(Z, noise_mag, alpha=2.2, beta=0.1, smooth_time=3, smooth_freq=5):
    # Z: complex STFT, noise_mag: 2D baseline magnitude (F x T or F x 1)
    mag = np.abs(Z)
    phase = np.angle(Z)

    # If noise_mag is F x 1, broadcast to frames
    if noise_mag.ndim == 1:
        noise_mag = noise_mag[:, None]

    # Over-subtraction with spectral floor
    clean_mag = mag - alpha * noise_mag
    floor = beta * mag
    clean_mag = np.maximum(clean_mag, floor)

    # Optional smoothing to reduce musical noise
    clean_mag_s = smooth_2d(clean_mag, tf=smooth_time, ff=smooth_freq)

    return clean_mag_s * np.exp(1j * phase)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", default=None, help="Input audio (defaults to Phase 2 output if present)")
    ap.add_argument("--noise_profile", default=None, help="npz from noise_profile.py")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Paths and ensure dirs
    outputs_dir = cfg["paths"]["outputs_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dirs([outputs_dir, logs_dir])

    # Resolve I/O
    default_in = os.path.join(outputs_dir, "phase2_wavelet.wav") if os.path.exists(os.path.join(outputs_dir, "phase2_wavelet.wav")) else cfg["paths"]["input"]
    in_path = args.input or default_in
    npz_default = os.path.join(outputs_dir, "phase1_noise_profile.npz")
    noise_npz = args.noise_profile or (npz_default if os.path.exists(npz_default) else None)
    if noise_npz is None:
        raise SystemExit("Noise profile npz not found. Run src/noise_profile.py first.")
    out_path = args.output or os.path.join(outputs_dir, "phase2b_spectral_subtract.wav")

    # Load audio
    x, sr = sf.read(in_path)
    if x.ndim > 1: x = np.mean(x, axis=1)

    # STFT config
    sc = cfg["spectral_subtraction"]
    n_fft = sc["n_fft"]; hop = sc["hop_length"]; win = n_fft
    window = get_window("hann", win)
    f, t, Z = stft(x, fs=sr, nperseg=win, noverlap=win-hop, window=window, boundary=None, padded=True, return_onesided=True)

    # Load noise profile
    prof = load_noise_profile(noise_npz)
    noise_mag = prof["baseline_mag"]  # F x Tn
    # If noise frames differ, use per-frequency baseline median
    if noise_mag.shape[0] != Z.shape[0]:
        # Simple rescale by trimming or interpolating along freq
        # For simplicity, take median across time and nearest resize along frequency bins
        nf = np.median(noise_mag, axis=1)
        # naive nearest resize
        nf_resized = np.interp(np.linspace(0, len(nf)-1, Z.shape[0]), np.arange(len(nf)), nf)
        noise_mag_use = nf_resized[:, None]
    else:
        # median over noise time to avoid leakage
        noise_mag_use = np.median(noise_mag, axis=1)[:, None]

    # Subtraction
    Z_clean = spectral_subtract(
        Z, noise_mag_use,
        alpha=sc["alpha"], beta=sc["beta"],
        smooth_time=sc["smoothing"]["time"],
        smooth_freq=sc["smoothing"]["freq"]
    )

    # ISTFT
    _, y = istft(Z_clean, fs=sr, nperseg=win, noverlap=win-hop, window=window, input_onesided=True)
    y = np.clip(y, -1.0, 1.0)
    sf.write(out_path, y, sr)

    # Log params
    with open(os.path.join(logs_dir, "phase2b_spectral_subtract.json"), "w") as fjson:
        json.dump({
            "input": in_path,
            "output": out_path,
            "noise_profile": noise_npz,
            "alpha": sc["alpha"], "beta": sc["beta"],
            "n_fft": n_fft, "hop_length": hop,
            "smoothing": sc["smoothing"]
        }, fjson, indent=2)

    print(f"Phase 2b (Spectral Subtraction) complete -> {out_path}")

if __name__ == "__main__":
    main()