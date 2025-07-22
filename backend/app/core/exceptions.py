"""
Custom exception classes and error handling.
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback

from app.core.logging import get_logger

logger = get_logger(__name__)


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass


class CustomerNotFoundError(Exception):
    """Raised when a customer is not found."""
    pass


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


class MLModelError(Exception):
    """Raised when ML model operations fail."""
    pass


class RAGSystemError(Exception):
    """Raised when RAG system operations fail."""
    pass


async def database_connection_exception_handler(request: Request, exc: DatabaseConnectionError):
    """Handle database connection errors."""
    logger.error(f"Database connection error: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "database_connection_error",
            "message": "Database is currently unavailable. Please try again later.",
            "detail": str(exc)
        }
    )


async def customer_not_found_exception_handler(request: Request, exc: CustomerNotFoundError):
    """Handle customer not found errors."""
    logger.warning(f"Customer not found: {exc}")
    return JSONResponse(
        status_code=404,
        content={
            "error": "customer_not_found",
            "message": "The requested customer was not found.",
            "detail": str(exc)
        }
    )


async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle custom validation errors."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "validation_error",
            "message": "The provided data is invalid.",
            "detail": str(exc)
        }
    )


async def ml_model_exception_handler(request: Request, exc: MLModelError):
    """Handle ML model errors."""
    logger.error(f"ML model error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "ml_model_error",
            "message": "Machine learning model is currently unavailable.",
            "detail": str(exc)
        }
    )


async def rag_system_exception_handler(request: Request, exc: RAGSystemError):
    """Handle RAG system errors."""
    logger.error(f"RAG system error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "rag_system_error",
            "message": "RAG system is currently unavailable.",
            "detail": str(exc)
        }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(f"Request validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "request_validation_error",
            "message": "The request data is invalid.",
            "details": exc.errors()
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
            "detail": "Internal server error"
        }
    )


def create_error_response(
    error_type: str,
    message: str,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """
    Create a standardized error response.
    
    Args:
        error_type: Type of error
        message: Human-readable error message
        status_code: HTTP status code
        details: Additional error details
        
    Returns:
        JSONResponse with error details
    """
    content = {
        "error": error_type,
        "message": message,
        "status_code": status_code
    }
    
    if details:
        content["details"] = details
    
    return JSONResponse(status_code=status_code, content=content)