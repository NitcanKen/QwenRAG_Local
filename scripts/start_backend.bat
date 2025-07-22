@echo off
echo Starting QwenRAG Backend Server...
echo.

REM Change to project directory
cd /d "C:\Users\asdzx\Documents\Python Project\QwenRAG_Local"

REM Check if Redis is running, start if not
echo Checking Redis server...
docker ps | findstr qwenrag-redis >nul
if %errorlevel% neq 0 (
    echo Starting Redis...
    docker start qwenrag-redis >nul 2>&1 || docker run -d --name qwenrag-redis -p 6379:6379 redis:7-alpine
    echo Waiting for Redis to start...
    timeout /t 3 >nul
) else (
    echo Redis is already running.
)

REM Check if Qdrant is running, start if not
echo Checking Qdrant vector database...
docker ps | findstr qdrant >nul
if %errorlevel% neq 0 (
    echo Starting Qdrant...
    docker run -d -p 6333:6333 -p 6334:6334 -v "%cd%\qdrant_storage:/qdrant/storage" --name qdrant-instance qdrant/qdrant
    echo Waiting for Qdrant to start...
    timeout /t 5 >nul
) else (
    echo Qdrant is already running.
)

REM Start backend server
echo.
echo Starting FastAPI backend server...
echo Server will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

cd backend
"%cd%\..\QwanRAG\Scripts\python.exe" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause