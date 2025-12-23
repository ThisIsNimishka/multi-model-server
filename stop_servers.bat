@echo off
REM ============================================================
REM Stop all Multi-Model AI Servers
REM ============================================================

echo Stopping all model servers...

REM Kill Python processes on specific ports
for %%p in (8000 8001 8002 8003 8004 8005) do (
    for /f "tokens=5" %%a in ('netstat -aon ^| findstr :%%p ^| findstr LISTENING') do (
        echo Stopping process on port %%p (PID: %%a)
        taskkill /PID %%a /F 2>NUL
    )
)

REM Alternative: Kill by window title
taskkill /FI "WINDOWTITLE eq Qwen*" /F 2>NUL
taskkill /FI "WINDOWTITLE eq Mistral*" /F 2>NUL
taskkill /FI "WINDOWTITLE eq Gemma*" /F 2>NUL
taskkill /FI "WINDOWTITLE eq API-Router*" /F 2>NUL

echo.
echo All servers stopped.
pause
