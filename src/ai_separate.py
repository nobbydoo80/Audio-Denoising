#!/usr/bin/env python
"""
Phase 4: AI-Based Enhancement (Optional)
Provides wrappers for Demucs, Spleeter, and other AI models
All AI features are disabled by default and require ai-extras.txt installation
"""
import argparse
import os
import json
import time
import subprocess
import sys
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf

def set_seed(seed: int):
    """Set deterministic seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    # Set seeds for AI frameworks if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def ensure_dirs(paths):
    """Create directories if they don't exist"""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def check_dependencies():
    """Check which AI dependencies are available"""
    deps = {
        "demucs": False,
        "spleeter": False,
        "speechbrain": False,
        "torch": False,
        "tensorflow": False
    }
    
    try:
        import demucs
        deps["demucs"] = True
    except ImportError:
        pass
    
    try:
        import spleeter
        deps["spleeter"] = True
    except ImportError:
        pass
    
    try:
        import speechbrain
        deps["speechbrain"] = True
    except ImportError:
        pass
    
    try:
        import torch
        deps["torch"] = True
    except ImportError:
        pass
    
    try:
        import tensorflow
        deps["tensorflow"] = True
    except ImportError:
        pass
    
    return deps

def apply_demucs(input_path, output_dir, model="htdemucs", stems=["vocals"], 
                 device="cpu", shifts=1, overlap=0.25, seed=42):
    """
    Apply Demucs source separation
    
    Returns path to separated vocal stem
    """
    try:
        import demucs.separate
        import torch
        
        # Set device
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available, using CPU")
            device = "cpu"
        
        # Prepare command
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", model,
            "--device", device,
            "--shifts", str(shifts),
            "--overlap", str(overlap),
            "-o", output_dir,
            "--seed", str(seed)
        ]
        
        # Add stems
        if stems:
            cmd.extend(["--two-stems", "vocals"])  # Simplified for vocals
        
        cmd.append(input_path)
        
        print(f"[INFO] Running Demucs: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[ERROR] Demucs failed: {result.stderr}")
            return None
        
        # Find output file
        model_dir = os.path.join(output_dir, model)
        input_name = Path(input_path).stem
        vocal_path = os.path.join(model_dir, input_name, "vocals.wav")
        
        if os.path.exists(vocal_path):
            return vocal_path
        else:
            print(f"[ERROR] Expected output not found: {vocal_path}")
            return None
            
    except ImportError:
        print("[WARN] Demucs not available. Install with: pip install demucs")
        return None
    except Exception as e:
        print(f"[ERROR] Demucs failed: {e}")
        return None

def apply_spleeter(input_path, output_dir, model="2stems", codec="wav", bitrate="128k"):
    """
    Apply Spleeter source separation
    
    Returns path to separated vocal stem
    """
    try:
        from spleeter.separator import Separator
        from spleeter.audio.adapter import AudioAdapter
        
        # Initialize separator
        separator = Separator(f'spleeter:{model}')
        audio_adapter = AudioAdapter.default()
        
        # Load audio
        waveform, sample_rate = audio_adapter.load(input_path)
        
        # Perform separation
        prediction = separator.separate(waveform)
        
        # Save vocals
        output_path = os.path.join(output_dir, "spleeter_vocals.wav")
        
        if 'vocals' in prediction:
            vocals = prediction['vocals']
            # Save using soundfile for consistency
            sf.write(output_path, vocals.T, sample_rate)
            return output_path
        else:
            print("[ERROR] No vocals in Spleeter output")
            return None
            
    except ImportError:
        print("[WARN] Spleeter not available. Install with: pip install spleeter")
        return None
    except Exception as e:
        print(f"[ERROR] Spleeter failed: {e}")
        return None

def apply_speech_enhancement(input_path, output_path, model="speechbrain/sepformer-wham",
                            device="cpu", chunk_size_sec=10, overlap_sec=1):
    """
    Apply SpeechBrain speech enhancement
    """
    try:
        from speechbrain.pretrained import SepformerSeparation as separator
        import torch
        import torchaudio
        
        # Load model
        print(f"[INFO] Loading SpeechBrain model: {model}")
        model = separator.from_hparams(source=model, savedir="pretrained_models",
                                       run_opts={"device": device})
        
        # Load audio
        waveform, sample_rate = torchaudio.load(input_path)
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Process in chunks if needed
        if waveform.shape[1] / sample_rate > chunk_size_sec:
            print(f"[INFO] Processing in {chunk_size_sec}s chunks...")
            chunk_samples = int(chunk_size_sec * sample_rate)
            overlap_samples = int(overlap_sec * sample_rate)
            
            output_chunks = []
            for i in range(0, waveform.shape[1], chunk_samples - overlap_samples):
                chunk = waveform[:, i:i+chunk_samples]
                est_sources = model.separate_batch(chunk)
                # Take first source (usually enhanced speech)
                output_chunks.append(est_sources[:, :, 0])
            
            # Concatenate with crossfade
            enhanced = torch.cat(output_chunks, dim=1)
        else:
            # Process entire file
            est_sources = model.separate_batch(waveform)
            enhanced = est_sources[:, :, 0]
        
        # Save
        torchaudio.save(output_path, enhanced, sample_rate)
        return output_path
        
    except ImportError:
        print("[WARN] SpeechBrain not available. Install with: pip install speechbrain")
        return None
    except Exception as e:
        print(f"[ERROR] Speech enhancement failed: {e}")
        return None

def apply_bandwidth_extension(input_path, output_path, target_sr=48000, method="sinc"):
    """
    Extend audio bandwidth using interpolation or neural methods
    """
    try:
        x, sr = sf.read(input_path)
        
        if sr >= target_sr:
            print(f"[INFO] Sample rate already {sr} Hz, no extension needed")
            sf.write(output_path, x, sr)
            return output_path
        
        if method == "sinc":
            # High-quality sinc interpolation
            from scipy import signal
            
            # Calculate resampling ratio
            ratio = target_sr / sr
            
            # Resample
            num_samples = int(len(x) * ratio)
            x_extended = signal.resample(x, num_samples)
            
            # Apply gentle high-frequency boost
            # Create filter for frequencies above original Nyquist
            nyquist_orig = sr / 2
            nyquist_new = target_sr / 2
            
            if nyquist_new > nyquist_orig:
                # Boost high frequencies slightly
                sos = signal.butter(2, [nyquist_orig * 0.9, nyquist_new * 0.9], 
                                   btype='band', fs=target_sr, output='sos')
                boost = signal.sosfilt(sos, x_extended) * 0.1
                x_extended = x_extended + boost
            
            # Normalize
            x_extended = np.clip(x_extended, -1.0, 1.0)
            
            sf.write(output_path, x_extended, target_sr)
            return output_path
            
        elif method == "neural":
            print("[WARN] Neural bandwidth extension not implemented, using sinc")
            return apply_bandwidth_extension(input_path, output_path, target_sr, "sinc")
            
    except Exception as e:
        print(f"[ERROR] Bandwidth extension failed: {e}")
        return None

def apply_dereverberation(input_path, output_path, room_size=0.5):
    """
    Apply dereverberation (simplified implementation)
    """
    try:
        x, sr = sf.read(input_path)
        
        # Simple spectral subtraction-based dereverb
        from scipy.signal import stft, istft
        
        # STFT
        f, t, Z = stft(x, fs=sr, nperseg=2048, noverlap=1536)
        S = np.abs(Z)
        phase = np.angle(Z)
        
        # Estimate reverb tail (simplified)
        # Assume reverb is in the decay portion
        reverb_estimate = np.percentile(S, 20, axis=1, keepdims=True)
        
        # Subtract scaled reverb estimate
        S_dereverb = S - room_size * reverb_estimate
        S_dereverb = np.maximum(S_dereverb, 0.1 * S)  # Floor to prevent over-suppression
        
        # Reconstruct
        Z_dereverb = S_dereverb * np.exp(1j * phase)
        _, x_dereverb = istft(Z_dereverb, fs=sr, nperseg=2048, noverlap=1536)
        
        # Normalize
        x_dereverb = np.clip(x_dereverb, -1.0, 1.0)
        
        sf.write(output_path, x_dereverb, sr)
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Dereverberation failed: {e}")
        return None

def classical_fallback(input_path, output_path, target="vocals"):
    """
    Classical DSP fallback when AI methods unavailable
    """
    print("[INFO] Using classical DSP fallback")
    
    x, sr = sf.read(input_path)
    
    if target == "vocals":
        # Simple vocal isolation using bandpass and center extraction
        from scipy import signal
        
        # Bandpass filter for vocal range (80-8000 Hz)
        sos = signal.butter(4, [80, 8000], btype='band', fs=sr, output='sos')
        x_filtered = signal.sosfilt(sos, x)
        
        # If stereo, extract center (vocals usually center-panned)
        if x.ndim > 1 and x.shape[1] == 2:
            # Mid-side processing
            mid = (x[:, 0] + x[:, 1]) / 2
            side = (x[:, 0] - x[:, 1]) / 2
            # Enhance mid, reduce side
            x_filtered = mid * 1.5 - side * 0.3
        
        # Normalize
        x_filtered = np.clip(x_filtered / (np.max(np.abs(x_filtered)) + 1e-10), -1.0, 1.0)
        
        sf.write(output_path, x_filtered, sr)
        return output_path
    
    # For other targets, just copy
    sf.write(output_path, x, sr)
    return output_path

def main():
    ap = argparse.ArgumentParser(description="Phase 4: AI Enhancement (Optional)")
    ap.add_argument("--config", required=True, help="Path to config file")
    ap.add_argument("--input", default=None, help="Input audio")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    
    # Load config
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Check if phase is enabled
    if not cfg.get("phases", {}).get("phase4", {}).get("enabled", False):
        print("[INFO] Phase 4 (AI Enhancement) is disabled in config, skipping")
        return
    
    # Set seed
    seed = cfg.get("ai_enhancement", {}).get("seed", cfg.get("global", {}).get("seed", 42))
    set_seed(seed)
    
    # Check dependencies
    deps = check_dependencies()
    print(f"[INFO] Available AI dependencies: {deps}")
    
    # Setup paths
    outputs_dir = cfg["paths"]["outputs_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dirs([outputs_dir, logs_dir])
    
    # Determine input
    prefer = [
        os.path.join(outputs_dir, "phase3_hpss.wav"),
        os.path.join(outputs_dir, "phase3_harmonic.wav"),
        os.path.join(outputs_dir, "phase2c_wiener.wav"),
        cfg["paths"]["input"]
    ]
    in_path = args.input or next((p for p in prefer if os.path.exists(p)), cfg["paths"]["input"])
    out_path = args.output or os.path.join(outputs_dir, "phase4_ai_enhanced.wav")
    
    # Get AI config
    ai_cfg = cfg.get("ai_enhancement", {})
    fallback = ai_cfg.get("fallback_classical", True)
    
    print(f"[INFO] Processing {in_path}")
    t0 = time.time()
    
    # Track what was applied
    applied_methods = []
    current_path = in_path
    temp_dir = os.path.join(outputs_dir, "ai_temp")
    ensure_dirs([temp_dir])
    
    # 1. Speech Enhancement
    if ai_cfg.get("speech_enhancement", {}).get("enabled", False):
        if deps["speechbrain"]:
            print("[INFO] Applying speech enhancement...")
            temp_out = os.path.join(temp_dir, "speech_enhanced.wav")
            result = apply_speech_enhancement(
                current_path, temp_out,
                model=ai_cfg["speech_enhancement"].get("model"),
                device=ai_cfg["speech_enhancement"].get("device", "cpu"),
                chunk_size_sec=ai_cfg["speech_enhancement"].get("chunk_size_sec", 10)
            )
            if result:
                current_path = result
                applied_methods.append("speech_enhancement")
        elif fallback:
            print("[INFO] SpeechBrain not available, using classical fallback")
            temp_out = os.path.join(temp_dir, "classical_enhanced.wav")
            result = classical_fallback(current_path, temp_out, "vocals")
            if result:
                current_path = result
                applied_methods.append("classical_vocal_isolation")
    
    # 2. Demucs Source Separation
    if ai_cfg.get("demucs", {}).get("enabled", False):
        if deps["demucs"]:
            print("[INFO] Applying Demucs source separation...")
            result = apply_demucs(
                current_path, temp_dir,
                model=ai_cfg["demucs"].get("model", "htdemucs"),
                stems=ai_cfg["demucs"].get("stems", ["vocals"]),
                device=ai_cfg["demucs"].get("device", "cpu"),
                shifts=ai_cfg["demucs"].get("shifts", 1),
                overlap=ai_cfg["demucs"].get("overlap", 0.25),
                seed=seed
            )
            if result:
                current_path = result
                applied_methods.append("demucs")
        elif fallback:
            print("[INFO] Demucs not available, using classical fallback")
            temp_out = os.path.join(temp_dir, "classical_separated.wav")
            result = classical_fallback(current_path, temp_out, "vocals")
            if result:
                current_path = result
                applied_methods.append("classical_separation")
    
    # 3. Spleeter Source Separation
    if ai_cfg.get("spleeter", {}).get("enabled", False) and "demucs" not in applied_methods:
        if deps["spleeter"]:
            print("[INFO] Applying Spleeter source separation...")
            result = apply_spleeter(
                current_path, temp_dir,
                model=ai_cfg["spleeter"].get("model", "2stems"),
                codec=ai_cfg["spleeter"].get("codec", "wav")
            )
            if result:
                current_path = result
                applied_methods.append("spleeter")
    
    # 4. Bandwidth Extension
    if ai_cfg.get("bandwidth_extension", {}).get("enabled", False):
        print("[INFO] Applying bandwidth extension...")
        temp_out = os.path.join(temp_dir, "bandwidth_extended.wav")
        result = apply_bandwidth_extension(
            current_path, temp_out,
            target_sr=ai_cfg["bandwidth_extension"].get("target_sr", 48000),
            method=ai_cfg["bandwidth_extension"].get("method", "sinc")
        )
        if result:
            current_path = result
            applied_methods.append("bandwidth_extension")
    
    # 5. Dereverberation
    if ai_cfg.get("dereverb", {}).get("enabled", False):
        print("[INFO] Applying dereverberation...")
        temp_out = os.path.join(temp_dir, "dereverbed.wav")
        result = apply_dereverberation(
            current_path, temp_out,
            room_size=ai_cfg["dereverb"].get("room_size", 0.5)
        )
        if result:
            current_path = result
            applied_methods.append("dereverberation")
    
    # Copy final result
    if current_path != in_path:
        import shutil
        shutil.copy2(current_path, out_path)
        print(f"[INFO] Saved enhanced audio to {out_path}")
    else:
        print("[WARN] No AI enhancements were applied")
        # Copy input to output for pipeline continuity
        import shutil
        shutil.copy2(in_path, out_path)
    
    dt = time.time() - t0
    
    # Log results
    log_data = {
        "phase": "4_ai_enhancement",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "input": in_path,
        "output": out_path,
        "available_dependencies": deps,
        "applied_methods": applied_methods,
        "configuration": ai_cfg,
        "processing_time_sec": dt,
        "fallback_used": fallback and not all(deps.values())
    }
    
    # Add instructions for enabling AI
    if not any(deps.values()):
        log_data["instructions"] = {
            "message": "No AI dependencies found. To enable AI features:",
            "steps": [
                "1. Install optional dependencies: pip install -r ai-extras.txt",
                "2. For Demucs: pip install demucs",
                "3. For Spleeter: pip install spleeter",
                "4. For SpeechBrain: pip install speechbrain",
                "5. Enable desired features in config under 'ai_enhancement'",
                "6. Set phase4.enabled: true in config"
            ]
        }
    
    log_path = os.path.join(logs_dir, "phase4_ai_enhancement.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"[INFO] Phase 4 complete in {dt:.1f}s")
    print(f"[INFO] Applied methods: {', '.join(applied_methods) if applied_methods else 'none'}")

if __name__ == "__main__":
    main()