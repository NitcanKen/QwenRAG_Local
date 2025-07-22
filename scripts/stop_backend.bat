@echo off
echo Stopping QwenRAG Backend Services...
echo.

REM Find and kill processes using port 8000
echo Stopping FastAPI server on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    echo Killing process %%a
    taskkill /f /pid %%a 2>nul
)

REM Kill any remaining python processes that might be uvicorn
echo Stopping any remaining uvicorn processes...
taskkill /f /im python.exe /fi "MEMUSAGE gt 50000" 2>nul
taskkill /f /im uvicorn.exe 2>nul

REM Give processes time to terminate
timeout /t 2 >nul

REM Stop and remove Qdrant container
echo Stopping Qdrant vector database...
docker stop qdrant-instance 2>nul
docker rm qdrant-instance 2>nul

REM Verify port 8000 is free
echo.
netstat -ano | findstr :8000 >nul
if %errorlevel% equ 0 (
    echo Warning: Port 8000 may still be in use
) else (
    echo Port 8000 is now free
)

echo.
echo Backend services stopped successfully.
echo.
pause