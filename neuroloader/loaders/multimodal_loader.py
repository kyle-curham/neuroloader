"""Multimodal dataset handling module"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import pandas as pd
from urllib.parse import urljoin

from .base_loader import BaseDataset
from .eeg_loader import EEGDataset
from .mri_loader import MRIDataset, FMRIDataset
from ..utils import find_files_by_extension, load_json_file, parse_bids_filename
from .. import logger

# Use the package's centralized logger
multimodal_logger = logger.get_logger('loaders.multimodal')

class MultimodalDataset(BaseDataset):
    """Class for handling multimodal neuroimaging datasets from OpenNeuro.
    
    This class coordinates multiple modality-specific handlers to work with
    datasets containing different types of neuroimaging data (e.g., MRI, fMRI, EEG).
    
    Modality detection is entirely handled by the factory (in factory.py) and passed
    to this class as a required parameter. No modality detection happens within this class.
    
    Handlers for each modality are initialized directly during construction based on 
    the detected modalities passed from the factory. This ensures efficient resource
    usage as only the necessary handlers are created.
    
    Like other loaders, this class inherits the download functionality from BaseDataset.
    """
    
    def __init__(
        self, 
        dataset_id: str,
        available_modalities: Dict[str, bool],
        data_dir: Optional[Union[str, Path]] = None,
        version: str = "latest"
    ):
        """Initialize a multimodal dataset.
        
        Args:
            dataset_id: The unique identifier for the dataset on OpenNeuro
            available_modalities: Dictionary of detected modalities
            data_dir: Directory where data will be stored (default: ./data)
            version: Dataset version to use (default: "latest")
        """
        # Initialize the base dataset handler
        super().__init__(dataset_id, data_dir, version)
        
        # Store available modalities
        self.available_modalities = available_modalities.copy()
        multimodal_logger.info(f"Using modalities: {[m for m, v in self.available_modalities.items() if v]}")
        
        # Directly initialize handlers for available modalities
        self.eeg_handler = EEGDataset(self.dataset_id, self.data_dir, self.version) if self.available_modalities["eeg"] else None
        self.mri_handler = MRIDataset(self.dataset_id, self.data_dir, self.version) if self.available_modalities["mri"] else None
        self.fmri_handler = FMRIDataset(self.dataset_id, self.data_dir, self.version) if self.available_modalities["fmri"] else None
        
        multimodal_logger.info(f"Initialized MultimodalDataset with ID: {dataset_id}")
    
    def get_recording_files(self) -> List[Path]:
        """Get all recording files across all available modalities in the dataset."""
        all_files = []
        
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            multimodal_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return []
        
        # Gather files from all available modalities
        if self.available_modalities["eeg"] and self.eeg_handler:
            all_files.extend(self.eeg_handler.get_recording_files())
            
        if self.available_modalities["mri"] and self.mri_handler:
            all_files.extend(self.mri_handler.get_recording_files())
            
        if self.available_modalities["fmri"] and self.fmri_handler:
            all_files.extend(self.fmri_handler.get_functional_scans())
            
        return all_files
    
    def get_structural_scans(self, scan_type: str = "T1w") -> List[Path]:
        """Get all structural MRI scans of a specific type.
        
        Args:
            scan_type: Type of structural scan to find (e.g., "T1w", "T2w")
            
        Returns:
            List[Path]: List of paths to structural scan files
        """
        if not self.available_modalities["mri"] or self.mri_handler is None:
            multimodal_logger.warning("MRI modality not available in this dataset")
            return []
        
        return self.mri_handler.get_structural_scans(scan_type)
    
    def get_functional_scans(self) -> List[Path]:
        """Get all functional MRI scans in the dataset.
        
        Returns:
            List[Path]: List of paths to functional scan files
        """
        if not self.available_modalities["fmri"] or self.fmri_handler is None:
            multimodal_logger.warning("fMRI modality not available in this dataset")
            return []
        
        return self.fmri_handler.get_functional_scans()
    
    def get_events_dataframe(self, recording_file: Union[str, Path]) -> pd.DataFrame:
        """Get events for a recording file.
        
        Args:
            recording_file: Path to the recording file
            
        Returns:
            pd.DataFrame: DataFrame with events information
        """
        recording_file = Path(recording_file)
        
        # Try to determine modality from filename
        if recording_file.suffix in ['.set', '.edf', '.bdf', '.vhdr', '.cnt', '.eeg']:
            if self.eeg_handler is None:
                multimodal_logger.error("EEG handler not available but EEG file requested")
                return pd.DataFrame()
            return self.eeg_handler.get_events_dataframe(recording_file)
        elif recording_file.suffix in ['.nii', '.nii.gz', '.img', '.hdr']:
            # Check filename for keywords to distinguish MRI vs fMRI
            if any(keyword in recording_file.name.lower() for keyword in ['bold', 'func', 'task']):
                if self.fmri_handler is None:
                    multimodal_logger.error("fMRI handler not available but fMRI file requested")
                    return pd.DataFrame()
                return self.fmri_handler.get_events_dataframe(recording_file)
            else:
                multimodal_logger.warning(f"File appears to be an MRI file with no events: {recording_file}")
                return pd.DataFrame()
        else:
            multimodal_logger.error(f"Could not determine modality for file: {recording_file}")
            return pd.DataFrame()
    
    def load_recording(self, recording_file: Union[str, Path], preload: bool = False) -> Any:
        """Load an EEG recording file with MNE.
        
        This implementation routes to the appropriate handler based on the file type.
        
        Args:
            recording_file: Path to the recording file
            preload: Whether to preload data into memory
            
        Returns:
            Any: The loaded recording or None if loading fails
        """
        recording_file = Path(recording_file)
        
        # Try to determine modality from filename
        if recording_file.suffix in ['.set', '.edf', '.bdf', '.vhdr', '.cnt', '.eeg']:
            if self.eeg_handler is None:
                multimodal_logger.error("EEG handler not available but EEG file requested")
                return None
            return self.eeg_handler.load_recording(recording_file, preload=preload)
        elif recording_file.suffix in ['.nii', '.nii.gz', '.img', '.hdr']:
            # Check filename for keywords to distinguish MRI vs fMRI
            if any(keyword in recording_file.name.lower() for keyword in ['bold', 'func', 'task']):
                if self.fmri_handler is None:
                    multimodal_logger.error("fMRI handler not available but fMRI file requested")
                    return None
                return self.fmri_handler.load_scan(recording_file)
            else:
                if self.mri_handler is None:
                    multimodal_logger.error("MRI handler not available but MRI file requested")
                    return None
                return self.mri_handler.load_scan(recording_file)
        else:
            multimodal_logger.error(f"Could not determine modality for file: {recording_file}")
            return None
    
    def get_subject_ids(self) -> List[str]:
        """Get a list of all subject IDs in the dataset.
        
        Returns:
            List[str]: List of unique subject IDs
        """
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            multimodal_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return []
            
        # Gather files from all modalities
        all_files = self.get_recording_files()
        
        # Add specific MRI/fMRI files that might not be included in get_recording_files
        if self.available_modalities["mri"] and self.mri_handler:
            all_files.extend(self.mri_handler.get_structural_scans("T1w"))
            all_files.extend(self.mri_handler.get_structural_scans("T2w"))
            
        if self.available_modalities["fmri"] and self.fmri_handler:
            all_files.extend(self.fmri_handler.get_functional_scans())
        
        # Extract subject IDs from filenames using BIDS naming convention
        subject_ids = set()
        
        for file_path in all_files:
            parts = parse_bids_filename(file_path.name)
            if 'sub' in parts:
                subject_ids.add(parts['sub'])
        
        return sorted(list(subject_ids))
    
    def get_subject_files(self, subject_id: str) -> Dict[str, List[Path]]:
        """Get all files for a specific subject, organized by modality.
        
        Args:
            subject_id: Subject ID to get files for
            
        Returns:
            Dict[str, List[Path]]: Dictionary with modalities as keys and file lists as values
        """
        result = {
            "eeg": [],
            "mri": [],
            "fmri": []
        }
        
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            multimodal_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return result
        
        # Helper function to filter files by subject ID
        def filter_by_subject(files: List[Path]) -> List[Path]:
            return [
                f for f in files 
                if f'sub-{subject_id}' in f.name or f'/sub-{subject_id}/' in str(f)
            ]
        
        # Get files for each modality
        if self.available_modalities["eeg"] and self.eeg_handler:
            result["eeg"] = filter_by_subject(self.eeg_handler.get_recording_files())
            
        if self.available_modalities["mri"] and self.mri_handler:
            result["mri"] = filter_by_subject(self.mri_handler.get_recording_files())
            result["mri"].extend(filter_by_subject(self.mri_handler.get_structural_scans("T1w")))
            result["mri"].extend(filter_by_subject(self.mri_handler.get_structural_scans("T2w")))
            
        if self.available_modalities["fmri"] and self.fmri_handler:
            result["fmri"] = filter_by_subject(self.fmri_handler.get_functional_scans())
        
        return result
    
    def get_multimodal_runs(self) -> Dict[str, Dict[str, List[Path]]]:
        """Identify matching runs across modalities based on BIDS metadata.
        
        This method tries to match recording files across modalities that belong
        to the same experimental run (e.g., simultaneous EEG-fMRI).
        
        Returns:
            Dict[str, Dict[str, List[Path]]]: Dictionary of runs with modality files
        """
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            multimodal_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return {}
        
        # Get all subject IDs
        subject_ids = self.get_subject_ids()
        
        # Dictionary to store matching runs
        matching_runs = {}
        
        for subject_id in subject_ids:
            subject_files = self.get_subject_files(subject_id)
            
            # Extract run and task identifiers from filenames
            run_info = {}
            
            # Process each modality
            for modality, files in subject_files.items():
                for file_path in files:
                    parts = parse_bids_filename(file_path.name)
                    
                    # If this file has task and run information
                    if 'task' in parts:
                        task_name = parts['task']
                        run_id = parts.get('run', '1')  # Default to run-1 if not specified
                        
                        # Create a unique key for this task-run combination
                        run_key = f"{subject_id}_{task_name}_{run_id}"
                        
                        # Add to run_info
                        if run_key not in run_info:
                            run_info[run_key] = {"eeg": [], "mri": [], "fmri": []}
                            
                        run_info[run_key][modality].append(file_path)
            
            # Add to matching runs if there are multimodal recordings
            for run_key, modality_files in run_info.items():
                # Count modalities with files
                modality_count = sum(1 for files in modality_files.values() if files)
                
                # If this run has data from multiple modalities
                if modality_count > 1:
                    matching_runs[run_key] = modality_files
        
        multimodal_logger.info(f"Found {len(matching_runs)} multimodal recording runs")
        return matching_runs
    
    def describe(self) -> Dict:
        """Get a description of the dataset.
        
        Returns:
            Dict: Dictionary containing dataset metadata
        """
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "data_dir": str(self.data_dir / self.dataset_id),
            "modalities": {
                modality: available 
                for modality, available in self.available_modalities.items()
            },
            "subject_count": len(self.get_subject_ids()) if self.is_downloaded() else None
        } 