@echo off
echo ========================================
echo Audio Restoration Pipeline Setup
echo ========================================
echo.

REM Check if virtual environment exists
if exist .venv (
    echo Virtual environment found at .venv
) else (
    echo Creating virtual environment...
    python -m venv .venv
)

echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Upgrading pip, setuptools, and wheel first...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Installing numpy with pre-built wheel...
python -m pip install numpy==1.26.4

echo.
echo Installing core dependencies...
python -m pip install scipy soundfile PyYAML

echo.
echo Installing optional dependencies...
python -m pip install matplotlib librosa

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To activate the environment manually, run:
echo   .venv\Scripts\activate.bat
echo.
echo Then run the pipeline with:
echo   python src/pipeline_v2.py --config configs/full_no_ai.yaml --phases all
echo.
pause