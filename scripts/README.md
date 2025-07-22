# Backend Management Scripts

Easy-to-use scripts for starting and stopping the QwenRAG backend server.

## Quick Start

### Windows Command Prompt
```bash
# Start backend
scripts\start_backend.bat

# Stop backend  
scripts\stop_backend.bat
```

### PowerShell
```powershell
# Start backend
.\scripts\start_backend.ps1

# Stop backend
.\scripts\stop_backend.ps1
```

## What These Scripts Do

### Start Backend (`start_backend.*`)
1. **Checks Qdrant**: Verifies if Qdrant vector database is running
2. **Starts Qdrant**: If not running, starts Qdrant Docker container
3. **Starts FastAPI**: Launches the backend server with auto-reload
4. **Shows URLs**: Displays server and documentation URLs

**Server URLs:**
- API Server: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Interactive API: http://localhost:8000/redoc

### Stop Backend (`stop_backend.*`)
1. **Stops FastAPI**: Terminates all Python/uvicorn processes
2. **Stops Qdrant**: Stops and removes the Qdrant Docker container
3. **Clean Shutdown**: Ensures all services are properly terminated

## Prerequisites

- Docker installed and running
- Virtual environment created (`QwanRAG`)
- Backend dependencies installed
- Qdrant Docker image available

## Troubleshooting

### Port Already in Use
If you get "port already in use" errors:
```bash
# Run the stop script first
scripts\stop_backend.bat

# Then start again
scripts\start_backend.bat
```

### Docker Issues
```bash
# Check Docker is running
docker ps

# Pull Qdrant image if needed
docker pull qdrant/qdrant
```

### Missing Dependencies
```bash
# Install missing packages
QwanRAG\Scripts\python.exe -m pip install sse-starlette
```

## Notes

- Scripts automatically handle virtual environment activation
- Qdrant data persists in `qdrant_storage/` directory
- Backend runs in development mode with auto-reload
- Use Ctrl+C to stop the server gracefully