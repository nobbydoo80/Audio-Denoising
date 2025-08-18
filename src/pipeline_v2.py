#!/usr/bin/env python
"""
Enhanced Pipeline Orchestrator v2
Supports all phases 1-6, conditional execution, dry-run, strict mode, and run manifest
"""
import argparse
import subprocess
import sys
import os
import json
import time
import hashlib
import platform
from pathlib import Path
from datetime import datetime

def script_path(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)

def get_git_info():
    """Get current git commit and branch info"""
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], 
                                        stderr=subprocess.DEVNULL).decode().strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], 
                                       stderr=subprocess.DEVNULL).decode().strip()[:8]
        return {"branch": branch, "commit": commit}
    except:
        return {"branch": "unknown", "commit": "unknown"}

def get_system_info():
    """Get system information for provenance tracking"""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

def compute_file_hash(filepath):
    """Compute SHA256 hash of a file"""
    if not os.path.exists(filepath):
        return None
    
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def check_dependencies(phase, strict=False):
    """Check if dependencies for a phase are available"""
    deps_ok = True
    missing = []
    
    if phase == 3:  # HPSS
        try:
            import scipy
        except ImportError:
            missing.append("scipy")
            deps_ok = False
    
    elif phase == 4:  # AI
        # Check optional AI dependencies
        ai_deps = []
        try:
            import demucs
            ai_deps.append("demucs")
        except ImportError:
            pass
        
        try:
            import spleeter
            ai_deps.append("spleeter")
        except ImportError:
            pass
        
        try:
            import speechbrain
            ai_deps.append("speechbrain")
        except ImportError:
            pass
        
        if not ai_deps:
            print(f"[WARN] Phase 4: No AI dependencies found. Install with: pip install -r ai-extras.txt")
            if strict:
                deps_ok = False
                missing.extend(["demucs/spleeter/speechbrain (at least one)"])
    
    elif phase == 6:  # Mastering
        try:
            import pyloudnorm
        except ImportError:
            print(f"[WARN] Phase 6: pyloudnorm not found, using fallback LUFS calculation")
            # Not critical, has fallback
    
    elif phase == "metrics":
        try:
            import pesq
        except ImportError:
            print(f"[WARN] Metrics: pesq not found, PESQ scores will be unavailable")
        
        try:
            import pystoi
        except ImportError:
            print(f"[WARN] Metrics: pystoi not found, STOI scores will be unavailable")
    
    if not deps_ok and strict:
        raise RuntimeError(f"Missing required dependencies for phase {phase}: {', '.join(missing)}")
    
    return deps_ok

def run_step(cmd, log_path, dry_run=False):
    """Run a single pipeline step"""
    if dry_run:
        print(f"[DRY-RUN] Would execute: {' '.join(cmd)}")
        return {"status": "dry-run", "elapsed": 0, "returncode": 0}
    
    t0 = time.time()
    print(f"[RUN] {' '.join(cmd)}")
    
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    
    # Save log
    Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout or "")
        f.write(f"\n[elapsed_sec]: {dt:.3f}\n")
        f.write(f"[returncode]: {proc.returncode}\n")
    
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Step failed: {' '.join(cmd)} (see {log_path})")
    
    print(f"[OK] ({dt:.1f}s) -> {log_path}")
    
    return {"status": "success", "elapsed": dt, "returncode": proc.returncode}

def load_config(config_path):
    """Load and validate configuration"""
    import yaml
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Validate required fields
    required = ["paths", "phases"]
    for field in required:
        if field not in cfg:
            raise ValueError(f"Missing required config field: {field}")
    
    # Normalize noise path key for compatibility with configs using 'noise_sample'
    paths = cfg.get("paths", {})
    if "noise" not in paths and "noise_sample" in paths:
        paths["noise"] = paths["noise_sample"]
        cfg["paths"] = paths
    
    return cfg

def create_run_manifest(cfg, args, phase_results):
    """Create comprehensive run manifest"""
    manifest = {
        "version": cfg.get("version", "2.0.0"),
        "run_id": f"run_{int(time.time())}",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "configuration": {
            "config_file": args.config,
            "config_hash": compute_file_hash(args.config),
            "seed": cfg.get("global", {}).get("seed", 42),
            "phases_requested": args.phases.split(",") if args.phases else [],
            "dry_run": args.dry_run,
            "strict_mode": args.strict
        },
        "inputs": {
            "audio": {
                "path": cfg["paths"]["input"],
                "hash": compute_file_hash(cfg["paths"]["input"])
            },
            "noise": {
                "path": cfg["paths"]["noise"],
                "hash": compute_file_hash(cfg["paths"]["noise"])
            }
        },
        "system": get_system_info(),
        "git": get_git_info(),
        "phases": phase_results,
        "outputs": {},
        "metrics": {},
        "total_elapsed_sec": sum(r.get("elapsed", 0) for r in phase_results.values())
    }
    
    # Add output file info
    outputs_dir = cfg["paths"]["outputs_dir"]
    final_output = cfg["paths"].get("final_output")
    
    if final_output and os.path.exists(final_output):
        manifest["outputs"]["final"] = {
            "path": final_output,
            "hash": compute_file_hash(final_output),
            "size_bytes": os.path.getsize(final_output)
        }
    
    # Add metrics if available
    metrics_path = cfg["paths"].get("metrics_path")
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            manifest["metrics"] = json.load(f)
    
    return manifest

def main():
    ap = argparse.ArgumentParser(description="Audio Restoration Pipeline Orchestrator v2")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--input", default=None, help="Override input audio path")
    ap.add_argument("--noise", default=None, help="Override noise sample path")
    ap.add_argument("--reference", default=None, help="Reference audio for metrics")
    ap.add_argument("--phases", default=None, help="Comma list of phases (e.g., 1,2,3 or all)")
    ap.add_argument("--dry-run", action="store_true", help="Plan execution without running")
    ap.add_argument("--strict", action="store_true", help="Fail on missing dependencies")
    ap.add_argument("--profile", default=None, help="Mastering profile")
    args = ap.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Override paths if provided
    if args.input:
        cfg["paths"]["input"] = args.input
    if args.noise:
        cfg["paths"]["noise"] = args.noise
    if args.reference:
        cfg["paths"]["reference"] = args.reference
    
    # Check global settings
    dry_run = args.dry_run or cfg.get("global", {}).get("dry_run", False)
    strict = args.strict or cfg.get("global", {}).get("strict_mode", False)
    
    # Determine which phases to run
    if args.phases:
        if args.phases.lower() == "all":
            selected_phases = [1, 2, 3, 4, 5, 6, "metrics"]
        else:
            selected_phases = []
            for p in args.phases.split(","):
                p = p.strip()
                if p == "metrics":
                    selected_phases.append("metrics")
                else:
                    try:
                        selected_phases.append(int(p))
                    except ValueError:
                        print(f"[WARN] Invalid phase: {p}")
    else:
        # Use config to determine enabled phases
        selected_phases = []
        for i in range(1, 7):
            phase_key = f"phase{i}"
            if cfg.get("phases", {}).get(phase_key, {}).get("enabled", False):
                selected_phases.append(i)
        
        if cfg.get("metrics", {}).get("enabled", False):
            selected_phases.append("metrics")
    
    print(f"[INFO] Pipeline v2 starting")
    print(f"[INFO] Phases to run: {selected_phases}")
    print(f"[INFO] Mode: {'DRY-RUN' if dry_run else 'EXECUTE'}")
    print(f"[INFO] Strict mode: {strict}")
    
    # Create directories
    for key in ["outputs_dir", "logs_dir", "reports_dir", "plots_dir"]:
        if key in cfg["paths"]:
            Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    
    # Save resolved config
    resolved_config_path = os.path.join(cfg["paths"]["logs_dir"], "resolved_config.json")
    with open(resolved_config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    
    # Track results
    phase_results = {}
    
    # Phase 1: Analysis + Noise Profile
    if 1 in selected_phases:
        print("\n[PHASE 1] Analysis & Noise Profile")
        check_dependencies(1, strict)
        
        result = run_step(
            [sys.executable, script_path("analyze_audio.py"), "--config", args.config, "--input", cfg["paths"]["input"], "--noise", cfg["paths"]["noise"]],
            os.path.join(cfg["paths"]["logs_dir"], "phase1_analyze_stdout.log"),
            dry_run
        )
        phase_results["phase1_analyze"] = result
        
        result = run_step(
            [sys.executable, script_path("noise_profile.py"), "--config", args.config, "--noise", cfg["paths"]["noise"]],
            os.path.join(cfg["paths"]["logs_dir"], "phase1_noise_profile_stdout.log"),
            dry_run
        )
        phase_results["phase1_noise_profile"] = result
    
    # Phase 2: Primary Noise Reduction
    if 2 in selected_phases:
        print("\n[PHASE 2] Primary Noise Reduction")
        check_dependencies(2, strict)
        
        # 2a: Wavelet
        result = run_step(
            [sys.executable, script_path("wavelet_denoise.py"), "--config", args.config],
            os.path.join(cfg["paths"]["logs_dir"], "phase2_wavelet_stdout.log"),
            dry_run
        )
        phase_results["phase2_wavelet"] = result
        
        # 2b: Spectral Subtraction
        result = run_step(
            [sys.executable, script_path("spectral_subtract.py"), "--config", args.config],
            os.path.join(cfg["paths"]["logs_dir"], "phase2b_spectral_subtract_stdout.log"),
            dry_run
        )
        phase_results["phase2b_spectral_subtract"] = result
        
        # 2c: Wiener Filter
        result = run_step(
            [sys.executable, script_path("wiener_filter.py"), "--config", args.config],
            os.path.join(cfg["paths"]["logs_dir"], "phase2c_wiener_stdout.log"),
            dry_run
        )
        phase_results["phase2c_wiener"] = result
    
    # Phase 3: HPSS
    if 3 in selected_phases:
        print("\n[PHASE 3] Harmonic-Percussive Separation")
        check_dependencies(3, strict)
        
        result = run_step(
            [sys.executable, script_path("hpss_separate.py"), "--config", args.config],
            os.path.join(cfg["paths"]["logs_dir"], "phase3_hpss_stdout.log"),
            dry_run
        )
        phase_results["phase3_hpss"] = result
    
    # Phase 4: AI Enhancement
    if 4 in selected_phases:
        print("\n[PHASE 4] AI Enhancement (Optional)")
        deps_ok = check_dependencies(4, strict)
        
        if deps_ok or not strict:
            result = run_step(
                [sys.executable, script_path("ai_separate.py"), "--config", args.config],
                os.path.join(cfg["paths"]["logs_dir"], "phase4_ai_stdout.log"),
                dry_run
            )
            phase_results["phase4_ai"] = result
        else:
            print("[SKIP] Phase 4 skipped due to missing dependencies")
            phase_results["phase4_ai"] = {"status": "skipped", "reason": "missing_deps"}
    
    # Phase 5: Artifact Control
    if 5 in selected_phases:
        print("\n[PHASE 5] Artifact Control")
        check_dependencies(5, strict)
        
        result = run_step(
            [sys.executable, script_path("artifact_control.py"), "--config", args.config],
            os.path.join(cfg["paths"]["logs_dir"], "phase5_artifact_stdout.log"),
            dry_run
        )
        phase_results["phase5_artifact"] = result
    
    # Phase 6: Mastering
    if 6 in selected_phases:
        print("\n[PHASE 6] Mastering & Normalization")
        check_dependencies(6, strict)
        
        cmd = [sys.executable, script_path("mastering_chain.py"), "--config", args.config]
        if args.profile:
            cmd.extend(["--profile", args.profile])
        
        result = run_step(
            cmd,
            os.path.join(cfg["paths"]["logs_dir"], "phase6_mastering_stdout.log"),
            dry_run
        )
        phase_results["phase6_mastering"] = result
    
    # Metrics Evaluation
    if "metrics" in selected_phases:
        print("\n[METRICS] Quality Evaluation")
        check_dependencies("metrics", strict)
        
        cmd = [sys.executable, script_path("metrics_eval.py"), "--config", args.config]
        if args.reference:
            cmd.extend(["--reference", args.reference])
        
        result = run_step(
            cmd,
            os.path.join(cfg["paths"]["logs_dir"], "metrics_eval_stdout.log"),
            dry_run
        )
        phase_results["metrics"] = result
    
    # Create and save run manifest
    print("\n[INFO] Creating run manifest...")
    manifest = create_run_manifest(cfg, args, phase_results)
    
    manifest_path = cfg["paths"].get("manifest_path", 
                                     os.path.join(cfg["paths"]["reports_dir"], "run_manifest.json"))
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"[INFO] Run manifest saved to {manifest_path}")
    
    # Summary
    total_time = sum(r.get("elapsed", 0) for r in phase_results.values())
    successful = sum(1 for r in phase_results.values() if r.get("status") == "success")
    skipped = sum(1 for r in phase_results.values() if r.get("status") == "skipped")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total time: {total_time:.1f}s")
    print(f"Phases run: {successful}")
    print(f"Phases skipped: {skipped}")
    
    if not dry_run:
        final_output = cfg["paths"].get("final_output")
        if final_output and os.path.exists(final_output):
            print(f"Final output: {final_output}")
            size_mb = os.path.getsize(final_output) / (1024 * 1024)
            print(f"Output size: {size_mb:.2f} MB")
    
    print("="*60)

if __name__ == "__main__":
    main()