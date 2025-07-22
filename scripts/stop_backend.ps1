Write-Host "Stopping QwenRAG Backend Services..." -ForegroundColor Red
Write-Host ""

# Find and kill processes using port 8000
Write-Host "Finding processes using port 8000..." -ForegroundColor Yellow
$port8000Processes = netstat -ano | Select-String ":8000.*LISTENING" | ForEach-Object {
    $fields = $_ -split '\s+'
    $pid = $fields[-1]
    if ($pid -match '^\d+$') { $pid }
}

foreach ($pid in $port8000Processes) {
    Write-Host "Stopping process $pid on port 8000..." -ForegroundColor Yellow
    try {
        Stop-Process -Id $pid -Force -ErrorAction Stop
        Write-Host "Process $pid stopped successfully." -ForegroundColor Green
    }
    catch {
        Write-Host "Failed to stop process $pid : $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Kill any remaining large Python processes (likely uvicorn)
Write-Host "Stopping large Python processes (uvicorn)..." -ForegroundColor Yellow
Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {$_.WorkingSet -gt 50MB} | ForEach-Object {
    Write-Host "Stopping Python process $($_.Id) ($(($_.WorkingSet/1MB).ToString('F0'))MB)..." -ForegroundColor Yellow
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

# Wait for processes to terminate
Start-Sleep -Seconds 2

# Stop and remove Qdrant container
Write-Host "Stopping Qdrant vector database..." -ForegroundColor Yellow
docker stop qdrant-instance 2>$null | Out-Null
docker rm qdrant-instance 2>$null | Out-Null

# Verify port 8000 is free
Write-Host ""
$port8000Check = netstat -ano | Select-String ":8000.*LISTENING"
if ($port8000Check) {
    Write-Host "Warning: Port 8000 may still be in use" -ForegroundColor Yellow
} else {
    Write-Host "Port 8000 is now free" -ForegroundColor Green
}

Write-Host ""
Write-Host "Backend services stopped successfully." -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to continue"