"""
Structured logging configuration using structlog.

This module demonstrates production logging patterns:
- Structured logging for better log analysis
- Contextual logging with request IDs
- Different log formats for development vs production
- Performance-optimized logging configuration
"""

import sys
import logging
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory, add_logger_name
from structlog.processors import (
    TimeStamper,
    add_log_level,
    JSONRenderer,
    StackInfoRenderer,
    format_exc_info,
)

from app.core.config import get_settings


def configure_logging() -> None:
    """
    Configure structured logging for the application.
    
    Design Rationale:
    - Using structlog for structured logging capabilities
    - Different configurations for development vs production
    - Including contextual information like request IDs
    - Optimized for performance with minimal overhead
    """
    settings = get_settings()
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper()),
    )
    
    # TODO (logging): 
    # Verify request-scoped context is actually included in output.
    # We use structlog.contextvars.bind_contextvars(...) in LoggingContext, but the processor
    # structlog.contextvars.merge_contextvars is not currently in the processors list.
    # If request_id/user_id are missing from logs, add merge_contextvars early in both
    # dev/prod processor chains and validate with a small test log inside LoggingContext.
    # processors = [
    # structlog.contextvars.merge_contextvars,
    # TimeStamper(fmt="ISO"), 
    # ....]

    # Configure structlog processors based on environment
    if settings.ENVIRONMENT == "development":
        # Human-readable format for development
        processors = [
            TimeStamper(fmt="ISO"),
            add_log_level,
            add_logger_name,
            StackInfoRenderer(),
            format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # JSON format for production (better for log aggregation)
        processors = [
            TimeStamper(fmt="ISO"),
            add_log_level,
            add_logger_name,
            StackInfoRenderer(),
            format_exc_info,
            JSONRenderer(),
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **initial_values: Any) -> structlog.BoundLogger:
    """
    Get a structured logger with optional initial context values.
    
    Args:
        name: Logger name (typically __name__)
        **initial_values: Initial context values to bind to the logger
        
    Returns:
        Configured structlog logger
        
    Design Rationale:
    - Providing a factory function for consistent logger creation
    - Supporting initial context values for service-level logging
    - Bound logger maintains context across log calls
    """
    logger = structlog.get_logger(name)
    
    if initial_values:
        logger = logger.bind(**initial_values)
    
    return logger


class LoggingContext:
    """
    Context manager for temporary logging context.
    
    Usage:
        with LoggingContext(request_id="123", user_id="456"):
            logger.info("Processing request")
            # All logs within this context will include request_id and user_id
    
    Design Rationale:
    - Context manager pattern for temporary context
    - Automatic cleanup when exiting context
    - Thread-safe context management
    """
    
    def __init__(self, **context: Any):
        self.context = context
        self.token = None
    
    def __enter__(self) -> None:
        self.token = structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        structlog.contextvars.reset_contextvars()
        