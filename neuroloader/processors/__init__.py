"""Neuroimaging preprocessing module.

This module contains processors for different neuroimaging modalities,
implementing standard preprocessing pipelines for EEG, MRI, fMRI, and
other neuroimaging data types.
"""

from .base_processor import BaseProcessor
from .eeg_processor import EEGProcessor
from .mri_processor import MRIProcessor
from .fmri_processor import FMRIProcessor
from .pipeline import PreprocessingPipeline

__all__ = [
    'BaseProcessor',
    'EEGProcessor',
    'MRIProcessor',
    'FMRIProcessor',
    'PreprocessingPipeline',
] 