@echo off
echo ========================================
echo Audio Restoration Pipeline Setup (Windows CMD)
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
echo Installing core dependencies...
python -m pip install -r requirements.txt

echo.
echo Optional: install metrics extras (PESQ, STOI, LUFS)?
echo   pip install -r requirements-metrics.txt

echo Optional: install AI extras (PyTorch, Demucs, SpeechBrain)?
echo   pip install -r requirements-ai.txt

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To activate the environment manually, run:
echo   .venv\Scripts\activate.bat

echo Then run the pipeline with:
echo   python src\pipeline_v2.py --config configs\full_no_ai.yaml --phases all

echo.
pause