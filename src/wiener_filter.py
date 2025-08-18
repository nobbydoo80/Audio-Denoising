#!/usr/bin/env python
import argparse, os, json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, get_window, medfilt

def ensure_dirs(paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def energy_vad(x, sr, frame_ms=30, threshold_db=-40, hangover_frames=5):
    frame_len = int(sr * frame_ms / 1000)
    n_frames = max(1, int(np.ceil(len(x)/frame_len)))
    speech = np.zeros(n_frames, dtype=bool)
    eps = 1e-12
    for i in range(n_frames):
        s = i*frame_len
        e = min(len(x), s+frame_len)
        frame = x[s:e]
        if len(frame)==0:
            speech[i] = False
            continue
        rms = np.sqrt(np.mean(frame**2)+eps)
        db = 20*np.log10(rms+eps)
        speech[i] = db > threshold_db
    out = speech.copy()
    last_true = -999
    for i in range(n_frames):
        if speech[i]: last_true = i
        if i - last_true <= hangover_frames:
            out[i] = True
    return out, frame_len

def wiener_gain_estimate(noisy_mag2, noise_psd, speech_psd_est):
    # SNR estimate
    snr = speech_psd_est / (noise_psd + 1e-12)
    gain = snr / (1.0 + snr)
    return np.clip(gain, 0.0, 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", default=None, help="Input audio (defaults to spectral_subtract output if present)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    outputs_dir = cfg["paths"]["outputs_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dirs([outputs_dir, logs_dir])

    # Resolve input preference: if spectral_subtraction was run, use that, else wavelet output, else raw input
    prefer = [
        os.path.join(outputs_dir, "phase2b_spectral_subtract.wav"),
        os.path.join(outputs_dir, "phase2_wavelet.wav"),
        cfg["paths"]["input"]
    ]
    in_path = args.input or next((p for p in prefer if os.path.exists(p)), cfg["paths"]["input"])
    out_path = args.output or os.path.join(outputs_dir, "phase2c_wiener.wav")

    # Load
    x, sr = sf.read(in_path)
    if x.ndim > 1: x = np.mean(x, axis=1)

    sc = cfg["spectral_subtraction"]
    n_fft = sc["n_fft"]; hop = sc["hop_length"]; win = n_fft
    window = get_window("hann", win)
    f, t, Z = stft(x, fs=sr, nperseg=win, noverlap=win-hop, window=window, boundary=None, padded=True, return_onesided=True)

    # Magnitude-squared
    noisy_mag2 = np.abs(Z)**2

    # Initial noise PSD estimate using first N ms of the evil child sample head (proxy here: use lowest-energies frames)
    wc = cfg["wiener"]
    update_ms = wc.get("update_ms", 20)
    kernel_bins = wc.get("median_kernel_bins", 5)
    vad_guided = wc.get("vad_guided", True)

    # Simple noise estimate: 10th percentile over time per frequency bin
    noise_psd = np.percentile(noisy_mag2, 10, axis=1)
    noise_psd = medfilt(noise_psd, kernel_size=kernel_bins)
    noise_psd = np.maximum(noise_psd, 1e-12)

    # VAD-guided speech PSD estimate (rolling)
    if vad_guided:
        vad_mask, frame_len = energy_vad(x, sr, frame_ms=update_ms, threshold_db=-40, hangover_frames=5)
        # Map STFT frames to VAD frames (approx)
        frames = Z.shape[1]
        vad_per_stft = np.zeros(frames, dtype=bool)
        samples_per_stft = hop
        for i in range(frames):
            s = i * samples_per_stft
            idx = min(int(s / frame_len), len(vad_mask)-1)
            vad_per_stft[i] = vad_mask[idx]
        speech_psd_est = np.where(vad_per_stft[None, :], noisy_mag2, 0.0)
        # Average where speech
        denom = np.maximum(np.sum(vad_per_stft), 1)
        speech_psd_est = np.sum(speech_psd_est, axis=1) / denom
    else:
        # fallback: use overall median as speech estimate
        speech_psd_est = np.median(noisy_mag2, axis=1)

    # Median smoothing on estimates
    speech_psd_est = medfilt(speech_psd_est, kernel_size=kernel_bins)
    speech_psd_est = np.maximum(speech_psd_est, noise_psd)

    # Compute gain per frequency, then expand to time with slow variation
    gain_f = wiener_gain_estimate(noisy_mag2.mean(axis=1), noise_psd, speech_psd_est)
    # Smooth gain across frequency to avoid musical noise
    gain_f = medfilt(gain_f, kernel_size=kernel_bins)
    G = gain_f[:, None]

    # Apply gain with anti-pumping like temporal smoothing (simple one-pole)
    attack_ms = 5
    release_ms = 50
    attack_a = np.exp(-1.0 / max(1, int(sr * attack_ms/1000 / hop)))
    release_a = np.exp(-1.0 / max(1, int(sr * release_ms/1000 / hop)))
    G_t = np.copy(G)
    for i in range(1, G.shape[1]):
        a = attack_a if G[:, i] > G_t[:, i-1] else release_a
        G_t[:, i] = a * G_t[:, i-1] + (1 - a) * G[:, i]

    Z_clean = Z * G_t
    _, y = istft(Z_clean, fs=sr, nperseg=win, noverlap=win-hop, window=window, input_onesided=True)
    y = np.clip(y, -1.0, 1.0)
    sf.write(out_path, y, sr)

    with open(os.path.join(logs_dir, "phase2c_wiener.json"), "w") as fjson:
        json.dump({
            "input": in_path,
            "output": out_path,
            "n_fft": n_fft,
            "hop_length": hop,
            "median_kernel_bins": kernel_bins,
            "vad_guided": vad_guided,
            "attack_ms": attack_ms,
            "release_ms": release_ms
        }, fjson, indent=2)

    print(f"Phase 2c (Wiener) complete -> {out_path}")

if __name__ == "__main__":
    main()