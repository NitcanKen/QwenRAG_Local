"""
Main FastAPI application for the Telco Customer Churn Dashboard.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.database import init_database
from app.core.middleware import LoggingMiddleware, SecurityHeadersMiddleware
from app.services.background_tasks import background_task_service
from app.services.chat_session_manager import init_session_manager, cleanup_session_manager
from app.services.deepseek_chat_service import init_deepseek_service, cleanup_deepseek_service
from app.core.exceptions import (
    DatabaseConnectionError,
    CustomerNotFoundError,
    ValidationError,
    MLModelError,
    RAGSystemError,
    database_connection_exception_handler,
    customer_not_found_exception_handler,
    validation_exception_handler,
    ml_model_exception_handler,
    rag_system_exception_handler,
    http_exception_handler,
    validation_error_handler,
    general_exception_handler,
)
from app.api.api_v1.api import api_router

# Setup logging
setup_logging()

# Create FastAPI application
app = FastAPI(
    title="Telco Customer Churn Dashboard API",
    description="AI-powered customer churn analytics and RAG system",
    version="1.0.0",
    openapi_url="/openapi.json"
)

# Add custom middleware (order matters - first added is executed last)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(LoggingMiddleware)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Add exception handlers
app.add_exception_handler(DatabaseConnectionError, database_connection_exception_handler)
app.add_exception_handler(CustomerNotFoundError, customer_not_found_exception_handler)
app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(MLModelError, ml_model_exception_handler)
app.add_exception_handler(RAGSystemError, rag_system_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, general_exception_handler)

@app.on_event("startup")
async def startup_event():
    """Initialize database connections, background tasks, and chat services on startup."""
    await init_database()
    await background_task_service.start_background_tasks()
    
    # Initialize chat services
    await init_session_manager()
    await init_deepseek_service()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up background tasks and chat services on shutdown."""
    await background_task_service.stop_background_tasks()
    
    # Cleanup chat services
    await cleanup_session_manager()
    await cleanup_deepseek_service()

@app.get("/")
async def root():
    """Health check endpoint."""
    return JSONResponse(
        content={
            "message": "Telco Customer Churn Dashboard API",
            "status": "healthy",
            "version": "1.0.0"
        }
    )

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return JSONResponse(
        content={
            "status": "healthy",
            "api_version": "1.0.0",
            "environment": settings.ENVIRONMENT
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development"
    )