@echo off
REM ============================================================
REM Windows-Compatible Multi-Model Server Startup
REM Works without vLLM/uvloop
REM ============================================================

setlocal EnableDelayedExpansion

echo ============================================================
echo   Multi-Model AI Server (Windows Compatible)
echo ============================================================
echo.

REM Activate virtual environment
call D:\vllm_env\Scripts\activate

REM Check argument
set MODE=%1
if "%MODE%"=="" set MODE=mistral

echo Starting %MODE% server...
echo.

if "%MODE%"=="mistral" (
    echo [Mistral-7B] Starting on port 8001...
    set CUDA_VISIBLE_DEVICES=0,1
    python windows_server.py --model mistral --port 8001
)

if "%MODE%"=="qwen" (
    echo [Qwen-VL-7B] Starting on port 8001...
    set CUDA_VISIBLE_DEVICES=0,1
    python windows_server.py --model qwen-vl --port 8001
)

if "%MODE%"=="gemma" (
    echo [Gemma-7B] Starting on port 8001...
    set CUDA_VISIBLE_DEVICES=0,1
    python windows_server.py --model gemma --port 8001
)

if "%MODE%"=="all" (
    echo Starting all models in separate windows...
    
    start "Mistral-7B" cmd /k "call D:\vllm_env\Scripts\activate && set CUDA_VISIBLE_DEVICES=0,1 && python windows_server.py --model mistral --port 8001"
    
    timeout /t 60 /nobreak
    
    start "Qwen-VL-7B" cmd /k "call D:\vllm_env\Scripts\activate && set CUDA_VISIBLE_DEVICES=2,3 && python windows_server.py --model qwen-vl --port 8002"
    
    echo.
    echo Servers starting...
    echo   - Mistral: http://localhost:8001
    echo   - Qwen-VL: http://localhost:8002
    echo.
)

pause
