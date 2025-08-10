@echo off
echo ========================================
echo Audio Restoration Pipeline Dependencies
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
echo Installing required dependencies...
pip install numpy scipy soundfile pyyaml matplotlib librosa

echo.
echo Installing optional metric dependencies...
pip install pesq pystoi

echo.
echo ========================================
echo Basic dependencies installed!
echo ========================================
echo.
echo To install AI dependencies, run:
echo   pip install torch demucs spleeter speechbrain
echo.
echo Or use the setup_env.bat script for a complete setup.
echo.
pause