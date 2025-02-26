"""
Centralized logging system for the neuroloader package.

This module provides consistent logging functionality across all components
of the neuroloader package, including file and console logging with configurable
levels, formats, and output destinations.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Union, Tuple

# Default log format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Package logger - this is now defined in __init__.py
# Instead of creating a new logger, get a reference to the one in __init__.py
logger = logging.getLogger('neuroloader')

def setup_logging(
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_dir: Optional[Union[str, Path]] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    capture_warnings: bool = True,
    log_file_prefix: str = 'neuroloader',
    propagate_to_root: bool = True,
    force_flush: bool = True
) -> Tuple[Path, Path]:
    """
    Set up logging for the neuroloader package.
    
    Args:
        console_level: Logging level for console output
        file_level: Logging level for file output
        log_dir: Directory to store log files (default: ./logs)
        log_format: Format string for log messages
        capture_warnings: Whether to capture warnings through logging
        log_file_prefix: Prefix for log filenames
        propagate_to_root: Whether to propagate logs to the root logger
        force_flush: Whether to force flushing logs to disk immediately
        
    Returns:
        Tuple[Path, Path]: Paths to the main log file and error log file
    """
    # Set up the package logger
    logger.setLevel(logging.DEBUG)  # Capture all logs, handlers will filter
    
    # Clear any existing handlers to prevent duplicate logging
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create log formatter
    formatter = logging.Formatter(log_format)
    
    # Add console handler (only one)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # Set up file logging if requested
    log_file, error_log_file = None, None
    if log_dir is not None:
        # Create log directory
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filenames with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{log_file_prefix}_{timestamp}.log"
        error_log_file = log_dir / f"{log_file_prefix}_errors_{timestamp}.log"
        
        try:
            # Add file handler for all logs
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Add file handler specifically for errors and warnings
            error_handler = logging.FileHandler(error_log_file, mode='a')
            error_handler.setLevel(logging.WARNING)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)
            
            logger.info(f"Logging to file: {log_file}")
            logger.info(f"Error logging to: {error_log_file}")
        except Exception as e:
            logger.error(f"Failed to set up file logging: {e}")
            logger.warning("Continuing with console-only logging")
    
    # Capture warnings through the logging system
    if capture_warnings:
        logging.captureWarnings(True)
    
    # Log system information
    logger.info(f"Neuroloader logging initialized")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Operating system: {sys.platform}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Force immediate flush to disk if requested
    if force_flush and log_file is not None:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
    
    return log_file, error_log_file

def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger that inherits settings from the neuroloader logger.
    
    Args:
        name: Name of the logger (will be prefixed with 'neuroloader.')
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if not name.startswith('neuroloader.'):
        name = f'neuroloader.{name}'
    return logging.getLogger(name)

def log_function_call(logger: logging.Logger, func_name: str, *args, **kwargs) -> None:
    """
    Log a function call with its arguments for debugging.
    
    Args:
        logger: Logger to use
        func_name: Name of the function being called
        *args: Positional arguments to the function
        **kwargs: Keyword arguments to the function
    """
    args_str = ', '.join([repr(arg) for arg in args])
    kwargs_str = ', '.join([f"{k}={repr(v)}" for k, v in kwargs.items()])
    all_args = ', '.join(filter(None, [args_str, kwargs_str]))
    logger.debug(f"Calling {func_name}({all_args})")

def log_exception(logger: logging.Logger, exc: Exception, context: str = "") -> None:
    """
    Log an exception with context information.
    
    Args:
        logger: Logger to use
        exc: Exception to log
        context: Optional context description
    """
    if context:
        logger.exception(f"{context}: {str(exc)}")
    else:
        logger.exception(str(exc))

def log_step(logger: logging.Logger, step_name: str, level: int = logging.INFO) -> None:
    """
    Log the start of a processing step.
    
    Args:
        logger: Logger to use
        step_name: Name of the processing step
        level: Logging level for the message
    """
    logger.log(level, f"=== {step_name} ===")

def configure_progress_logging(enable: bool = True) -> None:
    """
    Configure logging for progress updates (e.g., from tqdm).
    
    Args:
        enable: Whether to enable progress logging
    """
    if enable:
        # No special configuration needed for now
        pass
    else:
        # Disable progress logging
        logger.info("Progress logging disabled")

class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom LoggerAdapter to add consistent context to log messages.
    
    This adapter allows adding metadata like dataset_id, subject_id, etc.
    to create more informative and traceable logs.
    """
    
    def process(self, msg, kwargs):
        """
        Process the log message by adding context from extra dict.
        
        Args:
            msg: Log message
            kwargs: Logging keyword arguments
            
        Returns:
            Tuple[str, Dict]: Processed message and kwargs
        """
        # Format: "[key1=value1, key2=value2] Original message"
        context_str = ", ".join(f"{k}={v}" for k, v in self.extra.items())
        return f"[{context_str}] {msg}", kwargs

