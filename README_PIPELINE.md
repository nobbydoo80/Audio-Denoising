# Audio Restoration Pipeline - Production-Grade System

A comprehensive 6-phase audio restoration pipeline for removing persistent background noise (specifically child vocalizations) from interview recordings, with deterministic processing, optional AI enhancement, and professional mastering.

## Features

- **6-Phase Processing Pipeline**:
  1. **Analysis & Noise Profiling**: Spectral analysis, VAD, child-band detection
  2. **Primary Noise Reduction**: Wavelet denoising, spectral subtraction, Wiener filtering
  3. **HPSS**: Harmonic-Percussive Source Separation
  4. **AI Enhancement** (Optional): Demucs, SpeechBrain integration
  5. **Artifact Control**: Click removal, musical noise suppression, phase coherence
  6. **Mastering**: LUFS normalization, true-peak limiting, multiband compression

- **Production Features**:
  - Deterministic seeding for reproducibility
  - Dry-run mode for planning
  - Strict mode for CI/CD
  - Comprehensive logging and metrics
  - Run manifest with provenance tracking
  - Multiple config profiles

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository>
cd Audio-Denoising

# Create/activate a Python 3.10+ virtualenv as you like

# Install core (classical DSP) dependencies only
pip install -r requirements.txt

# Optional: metrics (PESQ/STOI/LUFS)
pip install -r requirements-metrics.txt

# Optional: AI (Demucs/SpeechBrain). For CUDA wheels, follow pytorch.org
pip install -r requirements-ai.txt
```

### 2. Basic Usage

```bash
# Run core denoising (Phases 1-2 only)
make run-core INPUT=jock_itntvw.wav NOISE=evilhild_2.wav

# Run full pipeline without AI (Phases 1-3, 5-6)
make run-full

# Run complete pipeline with AI (all phases)
make run-ai
```

### 3. Using Python Directly

```bash
# Core pipeline
python src/pipeline_v2.py --config configs/core_only.yaml --phases 1,2

# Full pipeline
python src/pipeline_v2.py --config configs/full_no_ai.yaml --phases all

# With AI
python src/pipeline_v2.py --config configs/full_with_ai.yaml --phases all

# Dry run to test
python src/pipeline_v2.py --config configs/full_no_ai.yaml --phases all --dry-run
```

## Configuration

Three pre-configured profiles are provided:

### 1. `configs/core_only.yaml`
- Phases 1-2 only
- Basic wavelet denoising + spectral methods
- Fastest processing
- No external AI dependencies

### 2. `configs/full_no_ai.yaml`
- All phases except AI (Phase 4 disabled)
- HPSS, artifact control, mastering
- Quality metrics evaluation
- No AI dependencies required

### 3. `configs/full_with_ai.yaml`
- All phases including AI enhancement
- Demucs/SpeechBrain for source separation
- Requires `requirements-ai.txt` installation

### 4. `configs/full_pipeline.yaml`
- Complete configuration reference
- All parameters exposed
- Extensive comments
- Use as template for custom configs

## Pipeline Phases

### Phase 1: Analysis & Noise Profile
```python
# Analyzes audio characteristics
python src/analyze_audio.py --config <config>

# Learns noise profile from sample
python src/noise_profile.py --config <config>
```
- Generates spectrograms
- Detects child vocalization bands (300-600Hz, 1-3kHz, 3-4kHz)
- Creates time-varying noise profile

### Phase 2: Primary Noise Reduction
```python
# Wavelet denoising
python src/wavelet_denoise.py --config <config>

# Spectral subtraction
python src/spectral_subtract.py --config <config>

# Wiener filtering
python src/wiener_filter.py --config <config>
```
- VisuShrink wavelet thresholding
- Alpha/beta spectral subtraction
- VAD-guided Wiener filtering

### Phase 3: HPSS
```python
python src/hpss_separate.py --config <config>
```
- Separates harmonic (speech) from percussive (noise)
- Median filtering or librosa methods
- Preserves transients

### Phase 4: AI Enhancement (Optional)
```python
python src/ai_separate.py --config <config>
```
- **Demucs**: Neural source separation
- **SpeechBrain**: Speech enhancement
- Falls back to classical methods if unavailable

### Phase 5: Artifact Control
```python
python src/artifact_control.py --config <config>
```
- Click/pop removal
- Musical noise suppression
- Pre-echo control
- De-essing
- Phase coherence (Griffin-Lim)
- Safety limiting

### Phase 6: Mastering
```python
python src/mastering_chain.py --config <config> --profile streaming
```
Profiles available:
- **streaming**: -14 LUFS (Spotify/Apple Music)
- **podcast**: -16 LUFS
- **broadcast**: -23 LUFS (EBU R128)
- **club**: -9 LUFS

Features:
- LUFS normalization
- True-peak limiting
- Multiband compression
- Tilt EQ
- Dithering with noise shaping

### Metrics Evaluation
```python
python src/metrics_eval.py --config <config> --reference clean.wav
```
Computes:
- SNR improvement
- PESQ (speech quality)
- STOI (intelligibility)
- SI-SDR
- Log-spectral distance

## Advanced Usage

### Dry Run Mode
Test pipeline without processing:
```bash
python src/pipeline_v2.py --config configs/full_no_ai.yaml --dry-run
```

### Strict Mode
Fail on missing dependencies (for CI/CD):
```bash
python src/pipeline_v2.py --config configs/full_no_ai.yaml --strict
```

### Custom Phase Selection
Run specific phases:
```bash
python src/pipeline_v2.py --config configs/full_no_ai.yaml --phases 1,3,5
```

## Output Structure

```
Audio-Denoising/
├── outputs/
│   ├── stages/
│   │   ├── phase1_noise_profile.npz
│   │   ├── phase2_wavelet.wav
│   │   ├── phase2b_spectral_subtract.wav
│   │   ├── phase2c_wiener.wav
│   │   ├── phase3_harmonic.wav
│   │   ├── phase3_percussive.wav
│   │   ├── phase3_hpss.wav
│   │   ├── phase4_ai_enhanced.wav
│   │   ├── phase5_artifact_controlled.wav
│   │   └── final_mastered_streaming.wav
│   └── final_mastered.wav
├── reports/
│   ├── plots/
│   │   ├── phase1_interview_spectrogram.png
│   │   └── phase1_noise_spectrogram.png
│   ├── metrics.json
│   └── run_manifest.json
└── logs/
    ├── phase*.json          # Per-phase parameters and metrics
    ├── phase*_stdout.log    # Console output
    └── resolved_config.json # Final configuration used
```

## Metrics & Quality Targets

| Metric | Target | Description |
|--------|--------|-------------|
| SNR Improvement | >15 dB | Signal-to-noise ratio gain |
| PESQ | >3.5 | Perceptual speech quality (1-5) |
| STOI | >0.85 | Short-time objective intelligibility (0-1) |
| SI-SDR | >10 dB | Scale-invariant signal-to-distortion |
| LUFS | -16±0.5 | Integrated loudness (podcast) |
| True Peak | <-1 dBTP | Maximum true peak level |

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| ModuleNotFoundError: demucs | Install AI extras: `pip install -r ai-extras.txt` |
| PESQ/STOI unavailable | Install metrics: `pip install pesq pystoi` |
| Robotic/metallic voice | Reduce spectral_subtraction.alpha to 1.8 |
| Pumping/breathing | Increase wiener.release_ms to 100 |
| Musical noise | Enable artifact_control.musical_noise |
| Residual child noise | Increase spectral_subtraction.alpha to 2.5 |

### Performance Optimization

For faster processing:
1. Disable unused phases in config
2. Reduce HPSS iterations
3. Skip metrics evaluation during testing
4. Use `phase4.ai_enhancement.speech_enhancement.chunk_size_sec: 30`

For better quality:
1. Enable all artifact control features
2. Increase Griffin-Lim iterations to 64
3. Use Demucs with shifts=5
4. Enable multiband HPSS

### GPU Acceleration

For AI phases with CUDA:
```yaml
ai_enhancement:
  demucs:
    device: cuda
  speech_enhancement:
    device: cuda
```

## Dependencies

### Minimal (requirements.txt)
- numpy
- scipy  
- soundfile
- matplotlib
- PyWavelets
- tqdm
- pyyaml

### Optional AI (ai-extras.txt)
- librosa (HPSS)
- pyloudnorm (LUFS)
- pesq (metrics)
- pystoi (metrics)

### AI Models (separate install)
- demucs
- spleeter
- speechbrain
- torch (for GPU)

## API Example

```python
from denoise import AudioDeNoise

# Simple usage with existing module
audioDenoiser = AudioDeNoise(inputFile="interview.wav")
audioDenoiser.deNoise(outputFile="interview_denoised.wav")

# Or use the pipeline programmatically
import subprocess
result = subprocess.run([
    "python", "src/pipeline_v2.py",
    "--config", "configs/full_no_ai.yaml",
    "--input", "interview.wav",
    "--noise", "noise_sample.wav",
    "--phases", "all"
])
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests in `tests/`
4. Ensure `make lint` passes
5. Submit a pull request

## License

See LICENSE file in repository.

## Citation

If you use this pipeline in research, please cite:
```
@software{audio_restoration_pipeline,
  title = {Production-Grade Audio Restoration Pipeline},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/Audio-Denoising}
}
```

## Support

For issues and questions:
- Check the troubleshooting section
- Review logs in `logs/` directory
- Open an issue with:
  - Config file used
  - Error messages from logs
  - Sample audio (if possible)