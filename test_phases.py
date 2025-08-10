#!/usr/bin/env python
"""
Test script to verify individual phases of the audio restoration pipeline
Run this to test each phase in isolation before running the full pipeline
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if result.stdout:
                print("Output:", result.stdout[:500])  # First 500 chars
        else:
            print(f"❌ FAILED: {description}")
            print("Error:", result.stderr[:1000])  # First 1000 chars of error
        return result.returncode == 0
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def test_phase(phase_num, script_name, config_file="configs/full_no_ai.yaml"):
    """Test a single phase"""
    cmd = f"python src/{script_name} --config {config_file}"
    return run_command(cmd, f"Phase {phase_num}: {script_name}")

def main():
    parser = argparse.ArgumentParser(description="Test individual pipeline phases")
    parser.add_argument("--phase", type=int, help="Test specific phase (1-6)")
    parser.add_argument("--config", default="configs/full_no_ai.yaml", 
                       help="Config file to use")
    parser.add_argument("--all", action="store_true", 
                       help="Test all phases sequentially")
    args = parser.parse_args()
    
    # Phase definitions
    phases = {
        1: ("analyze_audio.py", "Audio Analysis"),
        2: ("noise_profile.py", "Noise Profile Learning"),
        3: ("hpss_separate.py", "HPSS Separation"),
        4: ("ai_separate.py", "AI Enhancement"),
        5: ("artifact_control.py", "Artifact Control"),
        6: ("mastering_chain.py", "Mastering Chain")
    }
    
    # Additional phase 2 sub-phases
    phase2_scripts = [
        ("wavelet_denoise.py", "Wavelet Denoising"),
        ("spectral_subtract.py", "Spectral Subtraction"),
        ("wiener_filter.py", "Wiener Filtering")
    ]
    
    results = {}
    
    if args.all:
        # Test all phases
        print("\n" + "="*60)
        print("TESTING ALL PHASES")
        print("="*60)
        
        # Phase 1
        success = test_phase(1, phases[1][0], args.config)
        results["Phase 1"] = success
        
        # Phase 2 (multiple scripts)
        if success:
            success = test_phase("2a", phases[2][0], args.config)
            results["Phase 2a"] = success
            
            for script, desc in phase2_scripts:
                if success:
                    success = test_phase(f"2-{script.split('.')[0]}", script, args.config)
                    results[f"Phase 2-{desc}"] = success
        
        # Phases 3-6
        for phase_num in range(3, 7):
            if success or phase_num == 3:  # Continue even if phase 2 fails
                success = test_phase(phase_num, phases[phase_num][0], args.config)
                results[f"Phase {phase_num}"] = success
        
        # Metrics evaluation
        if success:
            cmd = "python src/metrics_eval.py --config " + args.config
            success = run_command(cmd, "Metrics Evaluation")
            results["Metrics"] = success
        
    elif args.phase:
        # Test specific phase
        if args.phase == 2:
            # Test all phase 2 scripts
            success = test_phase("2a", phases[2][0], args.config)
            results["Phase 2a"] = success
            
            for script, desc in phase2_scripts:
                success = test_phase(f"2-{script.split('.')[0]}", script, args.config)
                results[f"Phase 2-{desc}"] = success
        elif args.phase in phases:
            success = test_phase(args.phase, phases[args.phase][0], args.config)
            results[f"Phase {args.phase}"] = success
        else:
            print(f"Invalid phase number: {args.phase}")
            return
    else:
        print("Please specify --phase N or --all")
        parser.print_help()
        return
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for phase, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{phase}: {status}")
    
    total = len(results)
    passed = sum(1 for s in results.values() if s)
    print(f"\nTotal: {passed}/{total} phases passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to run full pipeline:")
        print("  python src/pipeline_v2.py --config configs/full_no_ai.yaml --phases all")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("You can still run phases 1-2 with:")
        print("  python src/pipeline_v2.py --config configs/full_no_ai.yaml --phases 1,2")

if __name__ == "__main__":
    main()