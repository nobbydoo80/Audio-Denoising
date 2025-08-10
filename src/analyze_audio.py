#!/usr/bin/env python
import argparse, json, os, random
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import stft
from pathlib import Path

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def ensure_dirs(paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def save_plot(path, fig):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def compute_spectrogram(x, sr, n_fft, hop_length, win_length, window, center):
    f, t, Zxx = stft(x, fs=sr, nperseg=win_length, noverlap=win_length-hop_length, window=window, padded=center, return_onesided=True, boundary=None)
    return f, t, Zxx

def energy_vad(x, sr, frame_ms=30, threshold_db=-40, hangover_frames=5):
    frame_len = int(sr * frame_ms / 1000)
    if frame_len <= 0: frame_len = 1
    n_frames = max(1, int(np.ceil(len(x)/frame_len)))
    speech = np.zeros(n_frames, dtype=bool)
    eps = 1e-12
    for i in range(n_frames):
        s = i*frame_len
        e = min(len(x), s+frame_len)
        frame = x[s:e]
        rms = np.sqrt(np.mean(frame**2)+eps)
        db = 20*np.log10(rms+eps)
        speech[i] = db > threshold_db
    # hangover smoothing
    out = speech.copy()
    last_true = -999
    for i in range(n_frames):
        if speech[i]: last_true = i
        if i - last_true <= hangover_frames:
            out[i] = True
    return out, frame_len

def summarize_band_energy(f, Z, bands):
    mag = np.abs(Z)
    summaries = {}
    for name, rng in bands.items():
        lb, ub = rng
        mask = (f >= lb) & (f <= ub)
        band_mag = mag[mask, :]
        summaries[name] = {
            "mean_db": float(20*np.log10(np.mean(band_mag)+1e-12)),
            "p95_db": float(np.percentile(20*np.log10(band_mag+1e-12), 95)),
            "p50_db": float(np.percentile(20*np.log10(band_mag+1e-12), 50)),
        }
    return summaries

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", default=None)
    ap.add_argument("--noise", default=None)
    args = ap.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    input_path = args.input or cfg["paths"]["input"]
    noise_path = args.noise or cfg["paths"]["noise"]
    out_dir = cfg["paths"]["outputs_dir"]
    rep_dir = cfg["paths"]["reports_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dirs([out_dir, rep_dir, logs_dir])

    sr_cfg = cfg["analysis"].get("samplerate", None)
    stft_cfg = cfg["analysis"]["stft"]
    spec_cfg = cfg["analysis"]["spectrogram"]
    child_bands = cfg["analysis"]["child_bands_hz"]
    vad_cfg = cfg["analysis"]["vad"]

    x, sr_x = sf.read(input_path)
    n, sr_n = sf.read(noise_path)
    if sr_cfg and sr_x != sr_cfg:
        print(f"[WARN] Interview SR {sr_x} != config {sr_cfg}. Proceeding with {sr_x}.")
    if sr_x != sr_n:
        print(f"[WARN] Interview SR {sr_x} != noise SR {sr_n}. Proceeding independently.")

    # Mono fold-down if needed
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    if n.ndim > 1:
        n = np.mean(n, axis=1)

    # Spectrograms
    f_x, t_x, Zx = compute_spectrogram(
        x, sr_x,
        stft_cfg["n_fft"], stft_cfg["hop_length"],
        stft_cfg["win_length"], stft_cfg["window"], stft_cfg["center"]
    )
    f_n, t_n, Zn = compute_spectrogram(
        n, sr_n,
        stft_cfg["n_fft"], stft_cfg["hop_length"],
        stft_cfg["win_length"], stft_cfg["window"], stft_cfg["center"]
    )

    # VAD on interview
    vad_enabled = vad_cfg.get("enabled", True)
    vad_mask, frame_len = energy_vad(
        x, sr_x, vad_cfg.get("frame_ms", 30),
        vad_cfg.get("energy_threshold_db", -40),
        vad_cfg.get("hangover_frames", 5)
    ) if vad_enabled else (None, None)

    # Summaries in child bands
    band_summary_x = summarize_band_energy(f_x, Zx, child_bands)
    band_summary_n = summarize_band_energy(f_n, Zn, child_bands)

    # Save plots
    # Interview spectrogram
    fig1 = plt.figure(figsize=(10,4))
    Sx_db = 20*np.log10(np.abs(Zx)+1e-12)
    vmin = np.max(Sx_db) - spec_cfg["dynamic_range_db"]
    plt.pcolormesh(t_x, f_x, Sx_db, shading="gouraud", cmap=spec_cfg["cmap"], vmin=vmin, vmax=np.max(Sx_db))
    plt.colorbar(label="dB")
    plt.title("Interview Spectrogram")
    plt.xlabel("Time [s]"); plt.ylabel("Freq [Hz]")
    save_plot(os.path.join(rep_dir, "phase1_interview_spectrogram.png"), fig1)

    # Noise spectrogram
    fig2 = plt.figure(figsize=(10,4))
    Sn_db = 20*np.log10(np.abs(Zn)+1e-12)
    vmin_n = np.max(Sn_db) - spec_cfg["dynamic_range_db"]
    plt.pcolormesh(t_n, f_n, Sn_db, shading="gouraud", cmap=spec_cfg["cmap"], vmin=vmin_n, vmax=np.max(Sn_db))
    plt.colorbar(label="dB")
    plt.title("Noise Sample Spectrogram")
    plt.xlabel("Time [s]"); plt.ylabel("Freq [Hz]")
    save_plot(os.path.join(rep_dir, "phase1_noise_spectrogram.png"), fig2)

    # Save summaries and simple time-varying noise profile (mean magnitude over time)
    noise_profile_time = np.mean(np.abs(Zn), axis=0).tolist()
    report = {
        "seed": seed,
        "samplerate_interview": sr_x,
        "samplerate_noise": sr_n,
        "child_band_summary_interview_db": band_summary_x,
        "child_band_summary_noise_db": band_summary_n,
        "noise_profile_time_mean_mag": noise_profile_time,
        "vad_frames": int(len(x)/frame_len) if vad_enabled else 0
    }
    with open(os.path.join(cfg["paths"]["logs_dir"], "phase1_analysis.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("Phase 1 analysis complete. Plots in reports/plots and JSON in logs/.")

if __name__ == "__main__":
    main()