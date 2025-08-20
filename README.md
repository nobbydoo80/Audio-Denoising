# Audio De-noising

A simple yet very powerful noise remover and reducer built in python.
The noise removed by using Wavelet Transform.

Wavelets has been very powerful tool to decompose the audio signal into parts and apply thresholds to eliminate
unwanted signal like noise. The thresholding method is the most important in the process of Audio De nosing.

The thresholding used is VisuShrink method or the universal threshold introduce by Donoho

This repo uses `pywt`. I have a custom implementation of wavelet here [wavelets](https://github.com/AP-Atul/wavelets) & [wavelets-ext](https://github.com/AP-Atul/wavelets-ext) (cython speed up)


## Execution

* Install the dependencies
  `$ pip3 install -r requirements.txt`
* Use the denoise.py file

  ```python
  from denoise import AudioDeNoise 
  
  audioDenoiser = AudioDeNoise(inputFile="input.wav")
  audioDenoiser.deNoise(outputFile="input_denoised.wav")
  audioDenoiser.generateNoiseProfile(noiseFile="input_noise_profile.wav")
  ```


## Linux Quick Start (Pipeline)

* Create a venv (Python 3.10 recommended for widest compatibility):

  ```bash
  python3.10 -m venv /home/you/Audio-Denoising/.venv_py310
  /home/you/Audio-Denoising/.venv_py310/bin/python -m pip install --upgrade pip
  /home/you/Audio-Denoising/.venv_py310/bin/python -m pip install -r requirements.txt pyyaml
  ```
* Minimal core DSP (no analysis/AI):

  ```bash
  make run-core-fast \
    PY=/home/you/Audio-Denoising/.venv_py310/bin/python \
    INPUT=/abs/path/to/interview.wav \
    NOISE=/abs/path/to/noise_sample.wav
  ```

  Outputs (listen to cleaned):
  * `outputs/stages/phase2_wavelet.wav`
  * `outputs/stages/phase2b_spectral_subtract.wav`
  * `outputs/stages/phase2c_wiener.wav` ← cleaned
* Full pipeline phases (Linux-safe):

  ```bash
  /home/you/Audio-Denoising/.venv_py310/bin/python \
    src/pipeline_v2.py --config configs/full_no_ai.yaml --phases 3,5,6
  ```

Notes:

* Configs may define `paths.noise_sample`; the pipeline accepts either and normalizes to `paths.noise`.
* Makefile is POSIX-only and uses bash; Windows paths were removed.
* If Phase 6 warns about LUFS, install: `pyloudnorm`.


