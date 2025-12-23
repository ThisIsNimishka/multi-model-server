@echo off
REM ============================================================
REM Installation Script for Multi-Model AI Server
REM ============================================================

echo ============================================================
echo Multi-Model AI Server - Installation Script
echo ============================================================
echo.

REM Check Python version
python --version 2>NUL
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or 3.11 from python.org
    pause
    exit /b 1
)

REM Check CUDA
nvcc --version 2>NUL
if errorlevel 1 (
    echo WARNING: CUDA toolkit not found in PATH
    echo vLLM requires CUDA. Make sure CUDA 12.x is installed.
    echo.
)

REM Set installation directory
set INSTALL_DIR=D:\vllm_env
set SCRIPT_DIR=%~dp0

echo Installation directory: %INSTALL_DIR%
echo.

REM Create virtual environment
if exist "%INSTALL_DIR%" (
    echo Virtual environment already exists.
    choice /C YN /M "Recreate it"
    if errorlevel 2 goto :skip_venv
    rmdir /s /q "%INSTALL_DIR%"
)

echo Creating virtual environment...
python -m venv %INSTALL_DIR%

:skip_venv

REM Activate virtual environment
call %INSTALL_DIR%\Scripts\activate

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA 12.8
echo.
echo Installing PyTorch with CUDA 12.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

REM Install vLLM
echo.
echo Installing vLLM (this may take a while)...
pip install vllm

REM Install other requirements
echo.
echo Installing additional requirements...
pip install fastapi uvicorn[standard] httpx transformers accelerate
pip install qwen-vl-utils tiktoken pydantic python-multipart

REM Verify installation
echo.
echo ============================================================
echo Verifying installation...
echo ============================================================

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
python -c "import vllm; print(f'vLLM installed successfully')"

echo.
echo ============================================================
echo Installation complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Copy start_servers.bat to your working directory
echo   2. Edit the model paths in start_servers.bat if needed
echo   3. Run: start_servers.bat
echo.
echo Virtual environment: %INSTALL_DIR%
echo Activate with: %INSTALL_DIR%\Scripts\activate
echo.

pause
