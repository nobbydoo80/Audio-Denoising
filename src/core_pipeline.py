#!/usr/bin/env python3
"""
Minimal Core DSP Pipeline
- Learns noise profile from provided noise sample
- Runs wavelet denoise on input
- Runs spectral subtraction using learned profile
- Runs Wiener filtering

Usage example:
  python src/core_pipeline.py \
    --config configs/core_only.yaml \
    --input /abs/path/interview.wav \
    --noise /abs/path/evil_child.wav
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path
import json
import time

THIS_DIR = Path(__file__).resolve().parent


def run_step(cmd: list[str], log_path: Path) -> None:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	print(f"[RUN] {' '.join(cmd)}")
	t0 = time.time()
	proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
	dt = time.time() - t0
	with open(log_path, "w", encoding="utf-8") as f:
		f.write(proc.stdout or "")
		f.write(f"\n[elapsed_sec]: {dt:.3f}\n")
		f.write(f"[returncode]: {proc.returncode}\n")
	if proc.returncode != 0:
		print(proc.stdout)
		raise SystemExit(f"Step failed: {' '.join(cmd)} (see {log_path})")
	print(f"[OK] ({dt:.1f}s) -> {log_path}")


def main() -> None:
	ap = argparse.ArgumentParser(description="Core DSP pipeline (no analysis, no AI)")
	ap.add_argument("--config", required=True)
	ap.add_argument("--input", required=True, help="Interview/input WAV path")
	ap.add_argument("--noise", required=True, help="Noise sample WAV path")
	args = ap.parse_args()

	# Prepare dirs from config
	import yaml
	with open(args.config, "r") as f:
		cfg = yaml.safe_load(f)

	# Normalize paths section
	paths = cfg.get("paths", {})
	paths["input"] = args.input
	# Support both keys but prefer --noise
	paths["noise"] = args.noise
	if "logs_dir" not in paths:
		paths["logs_dir"] = "logs"
	if "outputs_dir" not in paths:
		paths["outputs_dir"] = "outputs/stages"
	cfg["paths"] = paths

	logs_dir = Path(paths["logs_dir"]).resolve()
	outputs_dir = Path(paths["outputs_dir"]).resolve()
	logs_dir.mkdir(parents=True, exist_ok=True)
	outputs_dir.mkdir(parents=True, exist_ok=True)

	# Persist resolved config for provenance
	with open(logs_dir / "resolved_config_core.json", "w") as fjs:
		json.dump(cfg, fjs, indent=2)

	# 1) Learn noise profile
	run_step(
		[sys.executable, str(THIS_DIR / "noise_profile.py"), "--config", args.config, "--noise", args.noise],
		logs_dir / "core_phase1_noise_profile_stdout.log",
	)

	# 2) Wavelet denoise on input
	run_step(
		[sys.executable, str(THIS_DIR / "wavelet_denoise.py"), "--config", args.config, "--input", args.input],
		logs_dir / "core_phase2_wavelet_stdout.log",
	)

	# 3) Spectral subtraction (will pick wavelet output by default)
	run_step(
		[sys.executable, str(THIS_DIR / "spectral_subtract.py"), "--config", args.config],
		logs_dir / "core_phase2b_spectral_subtract_stdout.log",
	)

	# 4) Wiener filtering (prefers spectral subtract output)
	run_step(
		[sys.executable, str(THIS_DIR / "wiener_filter.py"), "--config", args.config],
		logs_dir / "core_phase2c_wiener_stdout.log",
	)

	print("\n[COMPLETE] Core DSP pipeline finished")
	print(f"Wavelet output: {outputs_dir / 'phase2_wavelet.wav'}")
	print(f"Spectral subtract output: {outputs_dir / 'phase2b_spectral_subtract.wav'}")
	print(f"Wiener output (cleaned): {outputs_dir / 'phase2c_wiener.wav'}")


if __name__ == "__main__":
	main() 