@echo off
REM ============================================================
REM Multi-Model AI Server Startup Script for Windows
REM ============================================================
REM 
REM GPU Allocation:
REM   GPU 0,1 (4080+3080) -> Qwen2.5-VL-7B (Port 8001)
REM   GPU 2,3 (2x 3080)   -> Mistral-7B    (Port 8002)
REM   GPU 4,5 (2x 3080)   -> Gemma-7B      (Port 8003) [Optional]
REM   GPU 6,7 (2x 3080)   -> Reserved
REM
REM Usage:
REM   start_servers.bat              - Start default models (Qwen + Mistral)
REM   start_servers.bat all          - Start all models
REM   start_servers.bat qwen         - Start only Qwen
REM   start_servers.bat mistral      - Start only Mistral
REM   start_servers.bat router       - Start only the API router
REM
REM ============================================================

setlocal EnableDelayedExpansion

REM Configuration
set MODELS_BASE=D:\AI_MODELS
set VENV_PATH=D:\vllm_env\Scripts\activate
set STARTUP_DELAY=45

REM Model paths
set QWEN_PATH=%MODELS_BASE%\models--Qwen--Qwen2.5-VL-7B-Instruct\snapshots\cc594898137f460bfe9f0759e9844b3ce807cfb5
set MISTRAL_PATH=%MODELS_BASE%\models--mistralai--Mistral-7B-Instruct-v0.2\snapshots\63a8b081895390a26e140280378bc85ec8bce07a
set GEMMA_PATH=%MODELS_BASE%\cache\models--google--gemma-7b-it\snapshots\9c5798d27f588501ce1e108079d2a19e4c3a2353

REM Check argument
set MODE=%1
if "%MODE%"=="" set MODE=default

echo ============================================================
echo Multi-Model AI Server Launcher
echo ============================================================
echo.
echo Mode: %MODE%
echo Models Base: %MODELS_BASE%
echo.

REM Activate virtual environment if exists
if exist "%VENV_PATH%" (
    echo Activating virtual environment...
    call %VENV_PATH%
) else (
    echo Warning: Virtual environment not found at %VENV_PATH%
    echo Using system Python...
)

REM Parse mode and start appropriate servers
if "%MODE%"=="all" goto :start_all
if "%MODE%"=="qwen" goto :start_qwen
if "%MODE%"=="mistral" goto :start_mistral
if "%MODE%"=="gemma" goto :start_gemma
if "%MODE%"=="router" goto :start_router
if "%MODE%"=="default" goto :start_default
goto :start_default

:start_default
echo Starting default configuration (Qwen + Mistral)...
call :launch_qwen
timeout /t %STARTUP_DELAY% /nobreak
call :launch_mistral
timeout /t 10 /nobreak
call :launch_router
goto :done

:start_all
echo Starting all models...
call :launch_qwen
timeout /t %STARTUP_DELAY% /nobreak
call :launch_mistral
timeout /t %STARTUP_DELAY% /nobreak
call :launch_gemma
timeout /t 10 /nobreak
call :launch_router
goto :done

:start_qwen
call :launch_qwen
goto :done

:start_mistral
call :launch_mistral
goto :done

:start_gemma
call :launch_gemma
goto :done

:start_router
call :launch_router
goto :done

REM ============================================================
REM Launch Functions
REM ============================================================

:launch_qwen
echo.
echo [Qwen2.5-VL-7B] Starting on GPUs 0,1 (Port 8001)...
if not exist "%QWEN_PATH%" (
    echo ERROR: Model path not found: %QWEN_PATH%
    exit /b 1
)
start "Qwen-VL-7B" cmd /k "set CUDA_VISIBLE_DEVICES=0,1 && python -m vllm.entrypoints.openai.api_server --model %QWEN_PATH% --port 8001 --tensor-parallel-size 2 --max-model-len 4096 --gpu-memory-utilization 0.85 --trust-remote-code"
echo Started Qwen-VL-7B server
exit /b 0

:launch_mistral
echo.
echo [Mistral-7B] Starting on GPUs 2,3 (Port 8002)...
if not exist "%MISTRAL_PATH%" (
    echo ERROR: Model path not found: %MISTRAL_PATH%
    exit /b 1
)
start "Mistral-7B" cmd /k "set CUDA_VISIBLE_DEVICES=2,3 && python -m vllm.entrypoints.openai.api_server --model %MISTRAL_PATH% --port 8002 --tensor-parallel-size 2 --max-model-len 8192 --gpu-memory-utilization 0.90"
echo Started Mistral-7B server
exit /b 0

:launch_gemma
echo.
echo [Gemma-7B] Starting on GPUs 4,5 (Port 8003)...
if not exist "%GEMMA_PATH%" (
    echo ERROR: Model path not found: %GEMMA_PATH%
    exit /b 1
)
start "Gemma-7B" cmd /k "set CUDA_VISIBLE_DEVICES=4,5 && python -m vllm.entrypoints.openai.api_server --model %GEMMA_PATH% --port 8003 --tensor-parallel-size 2 --max-model-len 8192 --gpu-memory-utilization 0.90"
echo Started Gemma-7B server
exit /b 0

:launch_router
echo.
echo [API Router] Starting on Port 8000...
start "API-Router" cmd /k "python -m uvicorn api_router:app --host 0.0.0.0 --port 8000"
echo Started API Router
exit /b 0

:done
echo.
echo ============================================================
echo Servers are starting up...
echo.
echo Endpoints:
echo   - Unified API:  http://localhost:8000/v1/chat/completions
echo   - Qwen-VL:      http://localhost:8001/v1/chat/completions
echo   - Mistral:      http://localhost:8002/v1/chat/completions
echo   - Gemma:        http://localhost:8003/v1/chat/completions
echo.
echo Health Check:    http://localhost:8000/health
echo Metrics:         http://localhost:8000/metrics
echo.
echo To stop all servers, close the terminal windows.
echo ============================================================

pause
