#!/usr/bin/env bash
set -euo pipefail
VENV_PY="/home/nobby/qzone/Audio-Denoising/.venv_py310/bin/python"
"${VENV_PY}" /home/nobby/qzone/Audio-Denoising/src/pipeline_v2.py \
  --config /home/nobby/qzone/Audio-Denoising/configs/core_only.yaml \
  --input /home/nobby/Downloads/interview_jock.wav \
  --noise /home/nobby/qzone/Audio-Denoising/src/evil_child.wav \
  --phases 1,2
