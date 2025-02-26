"""
Neuroimaging Data Manager Package

This package provides tools for downloading, processing, and analyzing neuroimaging 
data from OpenNeuro and other sources.
"""

from .base import BaseDataset
from .eeg import EEGDataset
from .mri import MRIDataset, FMRIDataset
from .multimodal import MultimodalDataset
from .factory import create_dataset
from .utils import download_file, validate_dataset

__version__ = "0.1.0" 