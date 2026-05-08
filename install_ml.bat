@echo off
title EPAM VERITAS - ML Dependencies Installation
color 0C
echo.
echo  ========================================
echo   EPAM VERITAS - ML Setup (Windows)
echo  ========================================
echo.
echo  This will install ML libraries for VERITAS.
echo  Requires: Python 3.11, NVIDIA CUDA 12.1+
echo  Takes 10-20 minutes depending on internet.
echo.
echo  Windows uses faster-whisper (Whisper large-v3)
echo  instead of NeMo Conformer for ASR.
echo.
pause

:: Check venv
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Creating venv first...
    python -m venv venv
)

:: Activate venv
call venv\Scripts\activate.bat

echo.
echo [1/6] Installing PyTorch with CUDA 12.1 support...
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    echo Make sure you have CUDA 12.1 drivers installed.
    pause
    exit /b 1
)

echo.
echo [2/6] Installing faster-whisper (ASR engine for Windows)...
pip install faster-whisper==1.1.1 soundfile
if errorlevel 1 (
    echo [WARNING] faster-whisper installation had issues.
    echo Trying without version pin...
    pip install faster-whisper soundfile
)

echo.
echo [3/6] Installing SpeechBrain (speaker diarization)...
pip install speechbrain==1.0.2 scikit-learn==1.6.1

echo.
echo [4/6] Installing HuggingFace transformers + quantization...
pip install transformers==4.48.3 accelerate==1.3.0
pip install bitsandbytes==0.45.1

echo.
echo [5/6] Installing remaining dependencies...
pip install -r requirements.txt --quiet

echo.
echo [6/6] Verifying installation...
echo.
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
python -c "from faster_whisper import WhisperModel; print('  faster-whisper OK')"
python -c "import speechbrain; print('  SpeechBrain OK')"
python -c "import transformers; print('  Transformers OK')"
python -c "import torchaudio; print('  torchaudio OK')"
python -c "import soundfile; print('  soundfile OK')"

echo.
echo  ========================================
echo   Installation complete!
echo.
echo   On Windows, VERITAS uses:
echo   - faster-whisper (Whisper large-v3) for ASR
echo   - torchaudio for audio preprocessing
echo   - SpeechBrain for speaker diarization
echo   - Qwen2.5-7B for summarization
echo.
echo   ffmpeg is OPTIONAL on Windows.
echo   Without it, torchaudio handles conversion.
echo.
echo   Next: run start_all.bat to start VERITAS
echo   First run downloads models (~6 GB total):
echo   - Whisper large-v3 ~3 GB
echo   - ECAPA-TDNN ~300 MB
echo   - Qwen2.5-7B ~4 GB (4-bit)
echo  ========================================
echo.
pause
