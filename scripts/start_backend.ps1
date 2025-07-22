Write-Host "Starting QwenRAG Backend Server..." -ForegroundColor Green
Write-Host ""

# Change to project directory
Set-Location "C:\Users\asdzx\Documents\Python Project\QwenRAG_Local"

# Check if Qdrant is running, start if not
Write-Host "Checking Qdrant vector database..." -ForegroundColor Yellow
$qdrantRunning = docker ps | Select-String "qdrant"
if (-not $qdrantRunning) {
    Write-Host "Starting Qdrant..." -ForegroundColor Yellow
    docker run -d -p 6333:6333 -p 6334:6334 -v "${PWD}\qdrant_storage:/qdrant/storage" --name qdrant-instance qdrant/qdrant
    Write-Host "Waiting for Qdrant to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
} else {
    Write-Host "Qdrant is already running." -ForegroundColor Green
}

# Start backend server
Write-Host ""
Write-Host "Starting FastAPI backend server..." -ForegroundColor Green
Write-Host "Server will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

Set-Location backend
& "..\QwanRAG\Scripts\python.exe" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000