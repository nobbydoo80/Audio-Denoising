# Audio Restoration Pipeline (Phases 1–2 Scaffold)

This directory contains initial scripts for Phases 1 and 2:
- Phase 1: Initial analysis and noise profile
- Phase 2: Primary noise reduction (wavelet, spectral subtraction, Wiener)

Run via pipeline:
python pipeline.py --config configs/default.yaml --input jock_itntvw.wav --noise evilhild_2.wav --phases 1,2

Outputs:
- outputs/stages/: Intermediate audio per stage
- reports/plots/: Spectrograms, distributions
- logs/: Per-stage logs