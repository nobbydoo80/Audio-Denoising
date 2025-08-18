#!/usr/bin/env python
import argparse
import subprocess
import sys
import os
from pathlib import Path
import json
import time

def run_step(cmd, log_path):
    t0 = time.time()
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout or "")
        f.write(f"\n[elapsed_sec]: {dt:.3f}\n")
        f.write(f"[returncode]: {proc.returncode}\n")
    if proc.returncode != 0:
        print(proc.stdout)
        raise SystemExit(f"Step failed: {' '.join(cmd)} (see {log_path})")
    print(f"[OK] ({dt:.1f}s) -> {log_path}")

def main():
    ap = argparse.ArgumentParser(description="Audio Restoration Pipeline Orchestrator (Phases 1–2)")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--input", default=None, help="Override input audio path")
    ap.add_argument("--noise", default=None, help="Override noise sample path")
    ap.add_argument("--phases", default="1,2", help="Comma list of phases to execute, e.g., 1,2 or 1")
    args = ap.parse_args()

    # Resolve paths
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.input:
        cfg["paths"]["input"] = args.input
    if args.noise:
        cfg["paths"]["noise"] = args.noise

    Path(cfg["paths"]["outputs_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["logs_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["reports_dir"]).mkdir(parents=True, exist_ok=True)

    # Persist possibly overridden cfg for reproducibility
    with open(os.path.join(cfg["paths"]["logs_dir"], "resolved_config.json"), "w") as fjs:
        json.dump(cfg, fjs, indent=2)

    selected = [p.strip() for p in args.phases.split(",") if p.strip()]
    # Phase 1: Initial analysis + noise profile
    if "1" in selected:
        run_step(
            [sys.executable, str(Path(__file__).resolve().parent / "analyze_audio.py"), "--config", args.config, "--input", cfg["paths"]["input"], "--noise", cfg["paths"]["noise"]],
            os.path.join(cfg["paths"]["logs_dir"], "phase1_analyze_stdout.log")
        )
        run_step(
            [sys.executable, str(Path(__file__).resolve().parent / "noise_profile.py"), "--config", args.config, "--noise", cfg["paths"]["noise"]],
            os.path.join(cfg["paths"]["logs_dir"], "phase1_noise_profile_stdout.log")
        )

    # Phase 2: Primary noise reduction (wavelet -> spectral subtraction -> Wiener)
    if "2" in selected:
        run_step(
            [sys.executable, str(Path(__file__).resolve().parent / "wavelet_denoise.py"), "--config", args.config, "--input", cfg["paths"]["input"]],
            os.path.join(cfg["paths"]["logs_dir"], "phase2_wavelet_stdout.log")
        )
        run_step(
            [sys.executable, str(Path(__file__).resolve().parent / "spectral_subtract.py"), "--config", args.config],
            os.path.join(cfg["paths"]["logs_dir"], "phase2b_spectral_subtract_stdout.log")
        )
        run_step(
            [sys.executable, str(Path(__file__).resolve().parent / "wiener_filter.py"), "--config", args.config],
            os.path.join(cfg["paths"]["logs_dir"], "phase2c_wiener_stdout.log")
        )

    print("Pipeline phases complete.")

if __name__ == "__main__":
    main()