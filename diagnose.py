#!/usr/bin/env python
"""
Diagnostic script to check pipeline readiness
"""
import os
import sys
import importlib.util
import yaml
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_import(module_name):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {module_name} is available")
        return True
    except ImportError:
        print(f"⚠️  {module_name} not available (optional)")
        return False

def check_script_syntax(script_path):
    """Check if a Python script has valid syntax"""
    try:
        with open(script_path, 'r') as f:
            compile(f.read(), script_path, 'exec')
        print(f"✅ Valid syntax: {script_path}")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {script_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking {script_path}: {e}")
        return False

def check_config(config_path):
    """Check if config file is valid YAML"""
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Check required sections
        required = ['paths', 'audio', 'phases', 'global']
        missing = [s for s in required if s not in cfg]
        
        if missing:
            print(f"⚠️  Config missing sections: {missing}")
            return False
        
        # Check paths
        if 'input' not in cfg['paths'] or 'noise_sample' not in cfg['paths']:
            print(f"❌ Config missing required paths (input/noise_sample)")
            return False
            
        print(f"✅ Valid config: {config_path}")
        return True
    except Exception as e:
        print(f"❌ Error in config {config_path}: {e}")
        return False

def main():
    print("="*60)
    print("AUDIO RESTORATION PIPELINE DIAGNOSTIC")
    print("="*60)
    
    all_good = True
    
    # Check input files
    print("\n📁 Input Files:")
    all_good &= check_file_exists("data/interview_jock.wav", "Interview audio")
    all_good &= check_file_exists("data/evil_child.wav", "Noise sample")
    
    # Check core scripts
    print("\n📜 Core Scripts:")
    scripts = [
        "src/analyze_audio.py",
        "src/noise_profile.py",
        "src/wavelet_denoise.py",
        "src/spectral_subtract.py",
        "src/wiener_filter.py",
        "src/hpss_separate.py",
        "src/ai_separate.py",
        "src/artifact_control.py",
        "src/mastering_chain.py",
        "src/metrics_eval.py",
        "src/pipeline_v2.py"
    ]
    
    for script in scripts:
        if check_file_exists(script, os.path.basename(script)):
            all_good &= check_script_syntax(script)
    
    # Check configs
    print("\n⚙️ Configuration Files:")
    configs = [
        "configs/full_no_ai.yaml",
        "configs/full_with_ai.yaml",
        "configs/core_only.yaml"
    ]
    
    for config in configs:
        if check_file_exists(config, os.path.basename(config)):
            all_good &= check_config(config)
    
    # Check required dependencies
    print("\n📦 Required Dependencies:")
    required_deps = [
        "numpy",
        "scipy",
        "soundfile",
        "pyyaml"
    ]
    
    for dep in required_deps:
        if not check_import(dep):
            all_good = False
    
    # Check optional dependencies
    print("\n📦 Optional Dependencies:")
    optional_deps = [
        "librosa",
        "matplotlib",
        "pyloudnorm",
        "pesq",
        "pystoi",
        "demucs",
        "spleeter",
        "speechbrain",
        "torch",
        "tensorflow"
    ]
    
    for dep in optional_deps:
        check_import(dep)
    
    # Check output directories
    print("\n📂 Output Directories:")
    dirs = ["outputs", "logs", "reports", "cache"]
    for d in dirs:
        exists = os.path.exists(d)
        if not exists:
            os.makedirs(d, exist_ok=True)
            print(f"✅ Created directory: {d}")
        else:
            print(f"✅ Directory exists: {d}")
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    if all_good:
        print("\n✅ All required components are ready!")
        print("\nYou can now run the pipeline with:")
        print("  python src/pipeline_v2.py --config configs/full_no_ai.yaml --phases all")
        print("\nOr test individual phases with:")
        print("  python test_phases.py --phase 3  # Test phase 3")
        print("  python test_phases.py --all       # Test all phases")
    else:
        print("\n⚠️ Some issues were found. Please address them before running.")
        print("\nCommon fixes:")
        print("1. Ensure input files are in data/ directory")
        print("2. Install required dependencies: pip install numpy scipy soundfile pyyaml")
        print("3. For optional features: pip install librosa matplotlib pyloudnorm")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())