"""Factory module for creating neuroimaging dataset handlers.

This module provides factory methods for creating the appropriate
dataset handler based on the contents of a neuroimaging dataset.
"""

from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import requests
import os
import json
from urllib.parse import urljoin

from .loaders.multimodal_loader import MultimodalDataset
from .loaders.mri_loader import MRIDataset, FMRIDataset
from .loaders.eeg_loader import EEGDataset
from .loaders.base_loader import BaseDataset
from .processors.pipeline import PreprocessingPipeline
from .processors.mri_processor import MRIProcessor
from .processors.fmri_processor import FMRIProcessor
from .processors.eeg_processor import EEGProcessor
from . import logger

# Use the package's centralized logger
factory_logger = logger.get_logger('factory')

def detect_modality_from_filelist(dataset_id: str, version: str = "latest") -> Dict[str, bool]:
    """
    Detect available modalities in a dataset without downloading it.
    
    Uses the GitHub API to get a list of files in the repository, then
    analyzes file patterns to determine what modalities are present.
    
    Args:
        dataset_id: The unique identifier for the dataset on OpenNeuro
        version: Dataset version to use (default: "latest")
        
    Returns:
        Dict[str, bool]: Dictionary of modalities and whether they are available
    """
    # Initialize modalities dict
    modalities = {
        "eeg": False,
        "mri": False,
        "fmri": False
    }
    
    # Construct API URL for the repository
    repo_url = f"https://api.github.com/repos/OpenNeuroDatasets/{dataset_id}/git/trees/"
    
    # If version is "latest", we need to get the main branch
    if version == "latest":
        try:
            # Get repository information to find the default branch
            repo_info_url = f"https://api.github.com/repos/OpenNeuroDatasets/{dataset_id}"
            response = requests.get(repo_info_url)
            response.raise_for_status()
            repo_info = response.json()
            default_branch = repo_info.get("default_branch", "main")
            version = default_branch
        except Exception as e:
            factory_logger.warning(f"Failed to get repository info: {str(e)}")
            version = "main"  # Fallback to "main" if we can't get the default branch
    
    # Get the tree for the specified version
    tree_url = f"{repo_url}{version}?recursive=1"
    
    try:
        # Fetch the file list from GitHub
        response = requests.get(tree_url)
        response.raise_for_status()
        tree_data = response.json()
        
        # Extract file paths
        files = [item["path"] for item in tree_data.get("tree", []) if item["type"] == "blob"]
        
        # Now analyze file patterns to determine modalities
        # Look for directory structure patterns first
        has_eeg_dir = any(path.startswith("sub-") and "/eeg/" in path for path in files)
        has_anat_dir = any(path.startswith("sub-") and "/anat/" in path for path in files)
        has_func_dir = any(path.startswith("sub-") and "/func/" in path for path in files)
        
        if has_eeg_dir:
            modalities["eeg"] = True
        if has_anat_dir:
            modalities["mri"] = True
        if has_func_dir:
            modalities["fmri"] = True
            
        # If no directories found, look for file extensions
        if not any(modalities.values()):
            # Define known file extensions for each modality
            eeg_exts = ['.set', '.edf', '.bdf', '.vhdr', '.cnt', '.eeg']
            mri_exts = ['.nii', '.nii.gz']
            
            for file_path in files:
                file_lower = file_path.lower()
                
                # Check EEG files
                if any(file_lower.endswith(ext) for ext in eeg_exts):
                    modalities["eeg"] = True
                
                # Check MRI/fMRI files
                if any(file_lower.endswith(ext) for ext in mri_exts):
                    if 'func/' in file_lower or 'bold' in file_lower or 'task' in file_lower:
                        modalities["fmri"] = True
                    elif 'anat/' in file_lower or any(tag in file_lower for tag in ['t1', 't2', 'mprage']):
                        modalities["mri"] = True
        
        factory_logger.info(f"Detected modalities without download: {[m for m, v in modalities.items() if v]}")
        return modalities
        
    except Exception as e:
        factory_logger.warning(f"Failed to detect modalities from file list: {str(e)}")
        # Return default (empty) modalities dictionary
        return modalities

def create_preprocessing_pipeline_for_modality(
    dataset: BaseDataset,
    modality: str,
    pipeline_options: Optional[Dict[str, Any]] = None
) -> Optional[PreprocessingPipeline]:
    """
    Create a preprocessing pipeline for a specific modality.
    
    Args:
        dataset: The dataset to create a pipeline for
        modality: The modality to create a pipeline for ('eeg', 'mri', or 'fmri')
        pipeline_options: Options for customizing the pipeline
        
    Returns:
        PreprocessingPipeline: The created preprocessing pipeline or None if not applicable
    """
    pipeline_options = pipeline_options or {}
    
    # Determine whether to skip preprocessing for derivative datasets
    skip_if_derivative = pipeline_options.get('skip_if_derivative', True)
    
    # Create a pipeline specific to the modality
    if modality == "eeg":
        pipeline = PreprocessingPipeline(name=f"{dataset.dataset_id}_eeg_pipeline", 
                                         skip_if_derivative=skip_if_derivative)
        
        # Create EEG processor
        eeg_processor = EEGProcessor(dataset)
        
        # Add EEG preprocessing steps
        pipeline.add_processor(
            processor=eeg_processor,
            name="eeg_preprocessing",
            params={
                "filtering": pipeline_options.get("eeg_filtering", True),
                "resampling": pipeline_options.get("eeg_resampling", True),
                "artifact_removal": pipeline_options.get("eeg_artifact_removal", True),
                "bad_channel_detection": pipeline_options.get("eeg_bad_channel_detection", True)
            },
            force_execution=pipeline_options.get("force_eeg_execution", False)
        )
        
        factory_logger.info(f"Created EEG preprocessing pipeline for {dataset.dataset_id}")
        return pipeline
        
    elif modality == "mri":
        pipeline = PreprocessingPipeline(name=f"{dataset.dataset_id}_mri_pipeline", 
                                         skip_if_derivative=skip_if_derivative)
        
        # Create MRI processor
        mri_processor = MRIProcessor(dataset)
        
        # Add MRI preprocessing steps
        pipeline.add_processor(
            processor=mri_processor,
            name="mri_preprocessing",
            params={
                "bias_correction": pipeline_options.get("mri_bias_correction", True),
                "skull_stripping": pipeline_options.get("mri_skull_stripping", True),
                "segmentation": pipeline_options.get("mri_segmentation", True),
                "normalization": pipeline_options.get("mri_normalization", True)
            },
            force_execution=pipeline_options.get("force_mri_execution", False)
        )
        
        factory_logger.info(f"Created MRI preprocessing pipeline for {dataset.dataset_id}")
        return pipeline
        
    elif modality == "fmri":
        pipeline = PreprocessingPipeline(name=f"{dataset.dataset_id}_fmri_pipeline", 
                                         skip_if_derivative=skip_if_derivative)
        
        # Create fMRI processor
        fmri_processor = FMRIProcessor(dataset)
        
        # Add fMRI preprocessing steps
        pipeline.add_processor(
            processor=fmri_processor,
            name="fmri_preprocessing",
            params={
                "motion_correction": pipeline_options.get("fmri_motion_correction", True),
                "slice_timing": pipeline_options.get("fmri_slice_timing", True),
                "spatial_smoothing": pipeline_options.get("fmri_spatial_smoothing", 6.0),
                "temporal_filtering": pipeline_options.get("fmri_temporal_filtering", True),
                "normalize": pipeline_options.get("fmri_normalize", True)
            },
            force_execution=pipeline_options.get("force_fmri_execution", False)
        )
        
        factory_logger.info(f"Created fMRI preprocessing pipeline for {dataset.dataset_id}")
        return pipeline
    
    factory_logger.warning(f"Unknown modality: {modality}")
    return None

def create_preprocessing_pipeline(
    dataset: BaseDataset,
    pipeline_options: Optional[Dict[str, Any]] = None
) -> Optional[PreprocessingPipeline]:
    """
    Create a preprocessing pipeline based on the dataset's modalities.
    
    Args:
        dataset: The dataset to create a pipeline for
        pipeline_options: Options for customizing the pipeline
        
    Returns:
        PreprocessingPipeline: The created preprocessing pipeline or None if not applicable
    """
    # Default options
    pipeline_options = pipeline_options or {}
    
    # Determine modalities available in the dataset
    if isinstance(dataset, MultimodalDataset):
        modalities = dataset.get_modalities()
        factory_logger.info(f"Creating pipeline for multimodal dataset with modalities: {modalities}")
        
        # For multimodal datasets, create a combined pipeline
        pipeline = PreprocessingPipeline(
            name=f"{dataset.dataset_id}_multimodal_pipeline",
            skip_if_derivative=pipeline_options.get('skip_if_derivative', True)
        )
        
        # Add appropriate processors based on available modalities
        if "eeg" in modalities:
            eeg_pipeline = create_preprocessing_pipeline_for_modality(
                dataset.get_modality_dataset("eeg"), "eeg", pipeline_options
            )
            if eeg_pipeline and eeg_pipeline.steps:
                # Add all steps from the EEG pipeline
                for step in eeg_pipeline.steps:
                    pipeline.steps.append(step)
                factory_logger.info("Added EEG preprocessing steps to multimodal pipeline")
        
        if "mri" in modalities:
            mri_pipeline = create_preprocessing_pipeline_for_modality(
                dataset.get_modality_dataset("mri"), "mri", pipeline_options
            )
            if mri_pipeline and mri_pipeline.steps:
                # Add all steps from the MRI pipeline
                for step in mri_pipeline.steps:
                    pipeline.steps.append(step)
                factory_logger.info("Added MRI preprocessing steps to multimodal pipeline")
        
        if "fmri" in modalities:
            fmri_pipeline = create_preprocessing_pipeline_for_modality(
                dataset.get_modality_dataset("fmri"), "fmri", pipeline_options
            )
            if fmri_pipeline and fmri_pipeline.steps:
                # Add all steps from the fMRI pipeline
                for step in fmri_pipeline.steps:
                    pipeline.steps.append(step)
                factory_logger.info("Added fMRI preprocessing steps to multimodal pipeline")
        
        # If pipeline has steps, return it; otherwise, return None
        if pipeline.steps:
            return pipeline
        else:
            factory_logger.warning("No preprocessing steps were added to the pipeline")
            return None
    
    # For single-modality datasets
    elif isinstance(dataset, EEGDataset):
        return create_preprocessing_pipeline_for_modality(dataset, "eeg", pipeline_options)
    elif isinstance(dataset, MRIDataset):
        return create_preprocessing_pipeline_for_modality(dataset, "mri", pipeline_options)
    elif isinstance(dataset, FMRIDataset):
        return create_preprocessing_pipeline_for_modality(dataset, "fmri", pipeline_options)
    else:
        factory_logger.warning(f"Unsupported dataset type: {type(dataset).__name__}")
        return None

def create_dataset(
    dataset_id: str,
    data_dir: Optional[Union[str, Path]] = None,
    version: str = "latest",
    force_type: Optional[str] = None,
    with_pipeline: bool = False,
    pipeline_options: Optional[Dict[str, Any]] = None
) -> Union[BaseDataset, Dict[str, Any]]:
    """
    Create the appropriate dataset handler based on available modalities.
    
    This factory function automatically detects the modalities available in the
    dataset and returns the appropriate specialized handler. If multiple modalities
    are detected, it returns a MultimodalDataset. If only one modality is found,
    it returns the appropriate specialized handler (EEGDataset, MRIDataset, or FMRIDataset).
    If no modalities are detected, it raises a ValueError.
    
    Optionally, it can create and return an appropriate preprocessing pipeline based on
    the detected modalities and derivative status.
    
    The factory will not download any files - this responsibility belongs to the dataset handlers.
    
    Args:
        dataset_id: The unique identifier for the dataset on OpenNeuro
        data_dir: Directory where data will be stored (default: ./data)
        version: Dataset version to use (default: "latest")
        force_type: Force a specific handler type regardless of content detection
                   (options: "multimodal", "eeg", "mri", "fmri")
        with_pipeline: Whether to also create and return a preprocessing pipeline
        pipeline_options: Options for customizing the preprocessing pipeline
    
    Returns:
        If with_pipeline is False:
            The appropriate dataset handler based on the detected modalities
        If with_pipeline is True:
            Dict with 'dataset' and 'pipeline' keys
        
    Raises:
        ValueError: If no modalities could be detected remotely
    """
    # If forcing a specific type, return that directly
    if force_type:
        if force_type.lower() == "multimodal":
            # Create a default modalities dictionary for forced multimodal type
            default_modalities = {"eeg": True, "mri": True, "fmri": True}
            dataset = MultimodalDataset(dataset_id, default_modalities, data_dir, version)
        elif force_type.lower() == "eeg":
            dataset = EEGDataset(dataset_id, data_dir, version)
        elif force_type.lower() == "mri":
            dataset = MRIDataset(dataset_id, data_dir, version)
        elif force_type.lower() == "fmri":
            dataset = FMRIDataset(dataset_id, data_dir, version)
        else:
            factory_logger.warning(
                f"Unknown force_type '{force_type}'. Using automatic detection instead."
            )
            dataset = None
    else:
        dataset = None
    
    # If we haven't created a dataset yet (no force_type or unknown force_type)
    if dataset is None:
        # First, attempt to detect modalities without downloading
        modalities = detect_modality_from_filelist(dataset_id, version)
        
        # Count available modalities
        available_modalities = [m for m, avail in modalities.items() if avail]
        modality_count = len(available_modalities)
        
        # If it's truly multimodal (has more than one modality), use MultimodalDataset
        if modality_count > 1:
            factory_logger.info(f"Detected multiple modalities without download: {', '.join(available_modalities)}")
            dataset = MultimodalDataset(dataset_id, modalities, data_dir, version)
        
        # If it's single-modality, use the appropriate specialized handler
        elif modality_count == 1:
            modality = available_modalities[0]
            factory_logger.info(f"Detected single modality without download: {modality}")
            
            if modality == "eeg":
                dataset = EEGDataset(dataset_id=dataset_id, data_dir=data_dir, version=version)
            
            elif modality == "fmri":
                dataset = FMRIDataset(dataset_id=dataset_id, data_dir=data_dir, version=version)
            
            elif modality == "mri":
                dataset = MRIDataset(dataset_id=dataset_id, data_dir=data_dir, version=version)
        
        # If no modalities were detected, raise an error
        else:
            error_msg = f"Could not detect any modalities for dataset '{dataset_id}'. Please check that the dataset ID is correct and the repository is accessible, or use force_type to specify the dataset type explicitly."
            factory_logger.error(error_msg)
            raise ValueError(error_msg)
    
    # If pipeline is requested, create and return it along with the dataset
    if with_pipeline:
        pipeline = create_preprocessing_pipeline(dataset, pipeline_options)
        return {
            "dataset": dataset,
            "pipeline": pipeline
        }
    
    # Otherwise just return the dataset
    return dataset 