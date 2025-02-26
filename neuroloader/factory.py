"""Factory module for creating neuroimaging dataset handlers.

This module provides factory methods for creating the appropriate
dataset handler based on the contents of a neuroimaging dataset.
"""

from pathlib import Path
from typing import Union, Optional, List

from .loaders.multimodal_loader import MultimodalDataset
from .loaders.mri_loader import MRIDataset, FMRIDataset
from .loaders.eeg_loader import EEGDataset
from .loaders.base_loader import BaseDataset
import logging

logger = logging.getLogger(__name__)

def create_dataset(
    dataset_id: str,
    data_dir: Optional[Union[str, Path]] = None,
    version: str = "latest",
    force_type: Optional[str] = None
) -> BaseDataset:
    """
    Create the appropriate dataset handler based on available modalities.
    
    This factory function automatically detects the modalities available in the
    dataset and returns the appropriate specialized handler. If multiple modalities
    are detected, it returns a MultimodalDataset. If only one modality is found,
    it returns the appropriate specialized handler (EEGDataset, MRIDataset, or FMRIDataset).
    
    Args:
        dataset_id: The unique identifier for the dataset on OpenNeuro
        data_dir: Directory where data will be stored (default: ./data)
        version: Dataset version to use (default: "latest")
        force_type: Force a specific handler type regardless of content detection
                   (options: "multimodal", "eeg", "mri", "fmri")
    
    Returns:
        The appropriate dataset handler based on the detected modalities
    """
    # If forcing a specific type, return that directly
    if force_type:
        if force_type.lower() == "multimodal":
            return MultimodalDataset(dataset_id, data_dir, version)
        elif force_type.lower() == "eeg":
            return EEGDataset(dataset_id, data_dir, version)
        elif force_type.lower() == "mri":
            return MRIDataset(dataset_id, data_dir, version)
        elif force_type.lower() == "fmri":
            return FMRIDataset(dataset_id, data_dir, version)
        else:
            logger.warning(
                f"Unknown force_type '{force_type}'. Using automatic detection instead."
            )
    
    # First initialize as multimodal to detect available modalities
    temp_dataset = MultimodalDataset(dataset_id=dataset_id, data_dir=data_dir, version=version)
    
    # If the dataset isn't already downloaded, download it
    # We need to download it to detect the modalities accurately
    if not temp_dataset.is_downloaded:
        success = temp_dataset.download_dataset()
        if not success:
            logger.warning(
                "Failed to download dataset. Using MultimodalDataset as a fallback."
            )
            return temp_dataset
    
    # Detect modalities
    temp_dataset._detect_modalities()
    
    # Count available modalities
    available_modalities = [m for m, avail in temp_dataset.available_modalities.items() if avail]
    modality_count = len(available_modalities)
    
    # If it's truly multimodal (has more than one modality), use MultimodalDataset
    if modality_count > 1:
        logger.info(f"Detected multiple modalities: {', '.join(available_modalities)}")
        return temp_dataset
    
    # If it's single-modality, use the appropriate specialized handler
    if modality_count == 1:
        modality = available_modalities[0]
        logger.info(f"Detected single modality: {modality}")
        
        if modality == "eeg":
            return EEGDataset(dataset_id=dataset_id, data_dir=data_dir, version=version)
        
        elif modality == "fmri":
            return FMRIDataset(dataset_id=dataset_id, data_dir=data_dir, version=version)
        
        elif modality == "mri":
            return MRIDataset(dataset_id=dataset_id, data_dir=data_dir, version=version)
    
    # If no modalities are detected, default to the multimodal handler
    logger.warning("No recognized modalities detected. Using MultimodalDataset as fallback.")
    return temp_dataset 