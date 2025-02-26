"""
Neuroimaging Data Loaders Package

This package provides loader classes for different neuroimaging modalities.
"""

from .base_loader import BaseDataset
from .eeg_loader import EEGDataset
from .mri_loader import MRIDataset, FMRIDataset
from .multimodal_loader import MultimodalDataset

__all__ = [
    'BaseDataset',
    'EEGDataset',
    'MRIDataset',
    'FMRIDataset',
    'MultimodalDataset'
] 