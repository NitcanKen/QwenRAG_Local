"""
Logging configuration for the application.
"""

import logging
import sys
from typing import Optional

from app.core.config import settings


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Set up application logging configuration.
    
    Args:
        log_level: Optional log level override
    """
    level = log_level or settings.LOG_LEVEL
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=settings.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log", mode="a", encoding="utf-8")
        ]
    )
    
    # Configure specific loggers
    loggers = [
        "uvicorn.access",
        "uvicorn.error",
        "fastapi",
        "sqlalchemy.engine",
        "celery",
        "app"
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)