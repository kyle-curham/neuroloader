"""Multimodal dataset handling module"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import pandas as pd
import logging
from urllib.parse import urljoin

from .base import BaseDataset
from .eeg import EEGDataset
from .mri import MRIDataset, FMRIDataset
from .utils import find_files_by_extension, load_json_file, parse_bids_filename

logger = logging.getLogger(__name__)

class MultimodalDataset:
    """Class for handling multimodal neuroimaging datasets from OpenNeuro.
    
    This class coordinates multiple modality-specific handlers to work with
    datasets containing different types of neuroimaging data (e.g., MRI, fMRI, EEG).
    """
    
    def __init__(
        self, 
        dataset_id: str,
        data_dir: Optional[Union[str, Path]] = None,
        version: str = "latest"
    ):
        """Initialize a multimodal dataset.
        
        Args:
            dataset_id: The unique identifier for the dataset on OpenNeuro
            data_dir: Directory where data will be stored (default: ./data)
            version: Dataset version to use (default: "latest")
        """
        self.dataset_id = dataset_id
        self.data_dir = Path(data_dir) if data_dir else Path("./data")
        self.version = version
        
        # Initialize modality-specific handlers
        self.eeg_handler = EEGDataset(dataset_id, data_dir, version)
        self.mri_handler = MRIDataset(dataset_id, data_dir, version)
        self.fmri_handler = FMRIDataset(dataset_id, data_dir, version)
        
        # Flag to track if dataset is downloaded
        self.is_downloaded = False
        
        # Metadata about available modalities
        self.available_modalities = {
            "eeg": False,
            "mri": False,
            "fmri": False
        }
        
        logger.info(f"Initialized MultimodalDataset with ID: {dataset_id}")
    
    def download_dataset(self, force: bool = False) -> bool:
        """Download the multimodal dataset from OpenNeuro.
        
        This method uses one of the handlers to download the entire dataset
        and then identifies which modalities are available.
        
        Args:
            force: If True, re-download even if the data exists locally
            
        Returns:
            bool: True if download was successful
        """
        # Use MRI handler to download the dataset (could use any handler)
        success = self.mri_handler.download_dataset(force=force)
        
        if success:
            self.is_downloaded = True
            
            # Detect available modalities
            self._detect_modalities()
            
            logger.info(f"Successfully downloaded multimodal dataset {self.dataset_id}")
            logger.info(f"Available modalities: {[m for m, available in self.available_modalities.items() if available]}")
            
        return success
    
    def _detect_modalities(self) -> None:
        """Detect which modalities are available in the dataset."""
        # Check for EEG files
        eeg_files = self.eeg_handler.get_recording_files()
        self.available_modalities["eeg"] = len(eeg_files) > 0
        
        # Check for structural MRI files
        t1_scans = self.mri_handler.get_structural_scans("T1w")
        t2_scans = self.mri_handler.get_structural_scans("T2w")
        self.available_modalities["mri"] = len(t1_scans) > 0 or len(t2_scans) > 0
        
        # Check for functional MRI files
        fmri_scans = self.fmri_handler.get_functional_scans()
        self.available_modalities["fmri"] = len(fmri_scans) > 0
    
    def get_eeg_files(self) -> List[Path]:
        """Get all EEG recording files in the dataset.
        
        Returns:
            List[Path]: List of paths to EEG recording files
        """
        if not self.available_modalities["eeg"]:
            logger.warning("EEG modality not available in this dataset")
            return []
        
        return self.eeg_handler.get_recording_files()
    
    def get_mri_files(self, scan_type: Optional[str] = None) -> List[Path]:
        """Get MRI scan files in the dataset.
        
        Args:
            scan_type: Optional filter for scan type (e.g., "T1w", "T2w")
            
        Returns:
            List[Path]: List of paths to MRI files
        """
        if not self.available_modalities["mri"]:
            logger.warning("MRI modality not available in this dataset")
            return []
        
        if scan_type:
            return self.mri_handler.get_structural_scans(scan_type)
        else:
            return self.mri_handler.get_recording_files()
    
    def get_fmri_files(self) -> List[Path]:
        """Get all fMRI recording files in the dataset.
        
        Returns:
            List[Path]: List of paths to fMRI recording files
        """
        if not self.available_modalities["fmri"]:
            logger.warning("fMRI modality not available in this dataset")
            return []
        
        return self.fmri_handler.get_functional_scans()
    
    def get_subject_ids(self) -> List[str]:
        """Get a list of all subject IDs in the dataset.
        
        Returns:
            List[str]: List of unique subject IDs
        """
        # Gather files from all modalities
        all_files = (
            self.get_eeg_files() + 
            self.get_mri_files() + 
            self.get_fmri_files()
        )
        
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
        
        # Helper function to filter files by subject ID
        def filter_by_subject(files: List[Path]) -> List[Path]:
            return [
                f for f in files 
                if f'sub-{subject_id}' in f.name or f'/sub-{subject_id}/' in str(f)
            ]
        
        # Get files for each modality
        if self.available_modalities["eeg"]:
            result["eeg"] = filter_by_subject(self.get_eeg_files())
            
        if self.available_modalities["mri"]:
            result["mri"] = filter_by_subject(self.get_mri_files())
            
        if self.available_modalities["fmri"]:
            result["fmri"] = filter_by_subject(self.get_fmri_files())
        
        return result
    
    def get_eeg_events(self, recording_file: Union[str, Path]) -> pd.DataFrame:
        """Get events for an EEG recording file.
        
        Args:
            recording_file: Path to the EEG recording file
            
        Returns:
            pd.DataFrame: DataFrame with events information
        """
        return self.eeg_handler.get_events_dataframe(recording_file)
    
    def get_fmri_events(self, recording_file: Union[str, Path]) -> pd.DataFrame:
        """Get events for an fMRI recording file.
        
        Args:
            recording_file: Path to the fMRI recording file
            
        Returns:
            pd.DataFrame: DataFrame with events information
        """
        return self.fmri_handler.get_events_dataframe(recording_file)
    
    def get_multimodal_runs(self) -> Dict[str, Dict[str, List[Path]]]:
        """Identify matching runs across modalities based on BIDS metadata.
        
        This method tries to match recording files across modalities that belong
        to the same experimental run (e.g., simultaneous EEG-fMRI).
        
        Returns:
            Dict[str, Dict[str, List[Path]]]: Dictionary of runs with modality files
        """
        if not self.is_downloaded:
            logger.warning("Dataset not yet downloaded. Call download_dataset() first.")
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
                    
                    # Skip files without task information
                    if 'task' not in parts:
                        continue
                    
                    task_name = parts['task']
                    run_id = parts.get('run', '1')  # Default to run-1 if not specified
                    
                    # Create a unique run identifier
                    run_key = f"sub-{subject_id}_task-{task_name}_run-{run_id}"
                    
                    # Add to run_info dictionary
                    if run_key not in run_info:
                        run_info[run_key] = {"eeg": [], "mri": [], "fmri": []}
                    
                    run_info[run_key][modality].append(file_path)
            
            # Add non-empty runs to the matching_runs dictionary
            for run_key, modality_files in run_info.items():
                # Only include runs that have files in at least two modalities
                modalities_present = sum(1 for files in modality_files.values() if files)
                if modalities_present >= 2:
                    matching_runs[run_key] = modality_files
        
        return matching_runs
    
    def describe(self) -> Dict:
        """Get a description of the multimodal dataset.
        
        Returns:
            Dict: Dictionary containing dataset metadata
        """
        # Ensure modalities are detected
        if self.is_downloaded and not any(self.available_modalities.values()):
            self._detect_modalities()
        
        # Get subject count
        subject_ids = self.get_subject_ids() if self.is_downloaded else []
        
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "data_dir": str(self.data_dir / self.dataset_id),
            "is_downloaded": self.is_downloaded,
            "available_modalities": {k: v for k, v in self.available_modalities.items() if v},
            "subject_count": len(subject_ids),
            "subjects": subject_ids
        } 