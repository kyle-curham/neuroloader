"""Neuroimaging data loading and processing package.

This package provides utilities for downloading, loading, and processing
neuroimaging datasets from OpenNeuro and other sources.
"""

# First, set up basic logging
import logging
import sys

# Create package logger but don't configure it yet - setup_logging will do that
logger = logging.getLogger('neuroloader')

# Now import components
from .factory import create_dataset
from .loaders.eeg_loader import EEGDataset
from .loaders.mri_loader import MRIDataset, FMRIDataset

# Set package version
__version__ = "0.1.0"

# Export key functions and classes
__all__ = [
    "create_dataset", 
    "EEGDataset", 
    "MRIDataset", 
    "FMRIDataset",
    "logger"
] 