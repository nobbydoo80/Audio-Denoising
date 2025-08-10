#!/usr/bin/env python
import argparse, os, json
from pathlib import Path
import soundfile as sf
import numpy as np

# Add project root to sys.path to import root-level modules when executed from src/
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# We will use repository's wavelet denoiser via import
# denoise.AudioDeNoise applies db4 wavelet with VisuShrink thresholding.
from denoise import AudioDeNoise

def ensure_dirs(paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    input_path = args.input or cfg["paths"]["input"]
    outputs_dir = cfg["paths"]["outputs_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dirs([outputs_dir, logs_dir])

    out_path = args.output or os.path.join(outputs_dir, "phase2_wavelet.wav")

    # Run wavelet denoiser
    dn = AudioDeNoise(inputFile=input_path)
    dn.deNoise(outputFile=out_path)

    # Log basic info
    meta = {}
    try:
        info = sf.info(out_path)
        meta = {
            "samplerate": info.samplerate,
            "channels": info.channels,
            "format": str(info.format)
        }
    except Exception as e:
        meta["error"] = str(e)

    with open(os.path.join(logs_dir, "phase2_wavelet.json"), "w") as f:
        json.dump({
            "input": input_path,
            "output": out_path,
            "wavelet_config_note": "Uses repository defaults (db4, level=2, VisuShrink soft). Exposed level in config but current denoise.py fixes level.",
            "meta": meta
        }, f, indent=2)

    print(f"Phase 2 (Wavelet) complete -> {out_path}")

if __name__ == "__main__":
    main()