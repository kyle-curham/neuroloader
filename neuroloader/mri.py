"""MRI dataset handling module"""

import os
import json
import requests
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
import logging
from urllib.parse import urljoin

from .base import BaseDataset
from .utils import find_files_by_extension, load_json_file, parse_bids_filename

logger = logging.getLogger(__name__)

class MRIDataset(BaseDataset):
    """Class for handling MRI datasets from OpenNeuro.
    
    This class provides methods for downloading, processing, and analyzing MRI data,
    particularly focusing on structural scans like T1 and T2.
    """
    
    def __init__(
        self, 
        dataset_id: str,
        data_dir: Optional[Union[str, Path]] = None,
        version: str = "latest"
    ):
        """Initialize an MRI dataset.
        
        Args:
            dataset_id: The unique identifier for the dataset on OpenNeuro
            data_dir: Directory where data will be stored (default: ./data)
            version: Dataset version to use (default: "latest")
        """
        super().__init__(dataset_id, data_dir, version)
        
        # MRI file extensions to search for
        self.mri_extensions = [
            '.nii', '.nii.gz', '.img', '.hdr'
        ]
        
        # Metadata file patterns
        self.metadata_file_patterns = [
            '*_T1w.json', '*_T2w.json', '*_bold.json', '*_dwi.json'
        ]
    
    def get_recording_files(self) -> List[Path]:
        """Get a list of MRI recording files in the dataset.
        
        Returns:
            List[Path]: List of paths to MRI files
        """
        return find_files_by_extension(self.dataset_dir, self.mri_extensions)
    
    def get_structural_scans(self, scan_type: str = "T1w") -> List[Path]:
        """Get all structural MRI scans of a specific type.
        
        Args:
            scan_type: Type of structural scan to find (e.g., "T1w", "T2w")
            
        Returns:
            List[Path]: List of paths to structural scan files
        """
        all_mri_files = self.get_recording_files()
        
        # Filter by scan type
        structural_scans = []
        for file_path in all_mri_files:
            if scan_type.lower() in file_path.name.lower():
                structural_scans.append(file_path)
        
        logger.info(f"Found {len(structural_scans)} {scan_type} structural scans")
        return structural_scans
    
    def load_scan(self, scan_file: Union[str, Path]) -> Tuple[Optional[nib.Nifti1Image], Optional[Dict]]:
        """Load an MRI scan file with nibabel and its metadata if available.
        
        Args:
            scan_file: Path to the scan file
            
        Returns:
            Tuple[Optional[nib.Nifti1Image], Optional[Dict]]: 
                Tuple of (NIfTI image object, metadata dictionary)
        """
        scan_file = Path(scan_file)
        
        if not scan_file.exists():
            logger.error(f"Scan file does not exist: {scan_file}")
            return None, None
        
        try:
            # Load the NIfTI file
            logger.info(f"Loading scan from {scan_file}")
            img = nib.load(scan_file)
            
            # Try to find metadata file
            metadata = None
            metadata_path = scan_file.with_suffix('.json')
            
            if metadata_path.exists():
                try:
                    metadata = load_json_file(metadata_path)
                    logger.info(f"Loaded metadata from {metadata_path}")
                except Exception as e:
                    logger.warning(f"Failed to load metadata: {str(e)}")
            
            return img, metadata
            
        except Exception as e:
            logger.error(f"Failed to load scan: {str(e)}")
            return None, None
    
    def get_events_dataframe(self, recording_file: Union[str, Path]) -> pd.DataFrame:
        """Create a DataFrame containing events for the specified recording.
        
        This is mostly applicable to fMRI data, but implemented here for 
        interface consistency.
        
        Args:
            recording_file: Path to the recording file
            
        Returns:
            pd.DataFrame: DataFrame with events information
        """
        recording_file = Path(recording_file)
        
        if not recording_file.exists():
            logger.error(f"Recording file does not exist: {recording_file}")
            return pd.DataFrame()
        
        # Try to find events file in the same directory
        events_file = recording_file.with_name(recording_file.stem + '_events.tsv')
        
        if not events_file.exists():
            # Try to find other events files in the same directory
            events_files = list(recording_file.parent.glob('*_events.tsv'))
            if events_files:
                events_file = events_files[0]
            else:
                logger.warning(f"No events file found for {recording_file}")
                return pd.DataFrame()
        
        try:
            # Load events from TSV file
            events_df = pd.read_csv(events_file, sep='\t')
            logger.info(f"Loaded events from {events_file}")
            return events_df
            
        except Exception as e:
            logger.error(f"Failed to load events file: {str(e)}")
            return pd.DataFrame()


class FMRIDataset(MRIDataset):
    """Class for handling fMRI datasets from OpenNeuro.
    
    This class extends MRIDataset with additional functionality specific to
    functional MRI data.
    """
    
    def __init__(
        self, 
        dataset_id: str,
        data_dir: Optional[Union[str, Path]] = None,
        version: str = "latest"
    ):
        """Initialize an fMRI dataset.
        
        Args:
            dataset_id: The unique identifier for the dataset on OpenNeuro
            data_dir: Directory where data will be stored (default: ./data)
            version: Dataset version to use (default: "latest")
        """
        super().__init__(dataset_id, data_dir, version)
        
        # Event file patterns specific to fMRI
        self.event_file_patterns = [
            '*_events.tsv', '*_bold.json'
        ]
    
    def get_functional_scans(self) -> List[Path]:
        """Get all functional MRI scans in the dataset.
        
        Returns:
            List[Path]: List of paths to functional scan files
        """
        all_mri_files = self.get_recording_files()
        
        # Filter for functional scans (containing 'bold', 'func', or 'task')
        functional_scans = []
        for file_path in all_mri_files:
            if any(keyword in file_path.name.lower() for keyword in ['bold', 'func', 'task']):
                functional_scans.append(file_path)
        
        logger.info(f"Found {len(functional_scans)} functional scans")
        return functional_scans
    
    def get_events_dataframe(self, recording_file: Union[str, Path]) -> pd.DataFrame:
        """Create a DataFrame containing events for the specified fMRI recording.
        
        This method overrides the base implementation with fMRI-specific event handling.
        
        Args:
            recording_file: Path to the recording file
            
        Returns:
            pd.DataFrame: DataFrame with events information
        """
        recording_file = Path(recording_file)
        
        if not recording_file.exists():
            logger.error(f"Recording file does not exist: {recording_file}")
            return pd.DataFrame()
        
        # Try to find events file based on BIDS naming convention
        parts = parse_bids_filename(recording_file.name)
        if 'sub' in parts and 'task' in parts:
            # Construct events filename following BIDS convention
            events_file_name = f"sub-{parts['sub']}_task-{parts['task']}_events.tsv"
            events_file = recording_file.parent / events_file_name
            
            if not events_file.exists():
                # Look for any events file in the directory
                events_files = list(recording_file.parent.glob('*_events.tsv'))
                if events_files:
                    events_file = events_files[0]
                else:
                    logger.warning(f"No events file found for {recording_file}")
                    return pd.DataFrame()
        else:
            # Fallback to simple name-based matching
            events_file = recording_file.with_name(recording_file.stem + '_events.tsv')
            
            if not events_file.exists():
                events_files = list(recording_file.parent.glob('*_events.tsv'))
                if events_files:
                    events_file = events_files[0]
                else:
                    logger.warning(f"No events file found for {recording_file}")
                    return pd.DataFrame()
        
        try:
            # Load events from TSV file
            events_df = pd.read_csv(events_file, sep='\t')
            
            # Ensure required columns exist
            if 'onset' not in events_df.columns:
                logger.error(f"Events file missing required 'onset' column: {events_file}")
                return pd.DataFrame()
            
            # Add duration column if missing
            if 'duration' not in events_df.columns:
                events_df['duration'] = 0.0
                logger.warning(f"Added default duration column to events")
            
            # Add trial_type column if missing
            if 'trial_type' not in events_df.columns and 'event_type' in events_df.columns:
                events_df['trial_type'] = events_df['event_type']
            elif 'trial_type' not in events_df.columns and 'condition' in events_df.columns:
                events_df['trial_type'] = events_df['condition']
            elif 'trial_type' not in events_df.columns:
                logger.warning(f"No trial_type column found in events file, using index as trial_type")
                events_df['trial_type'] = [f"event_{i}" for i in range(len(events_df))]
            
            logger.info(f"Loaded {len(events_df)} events from {events_file}")
            return events_df
            
        except Exception as e:
            logger.error(f"Failed to load events file: {str(e)}")
            return pd.DataFrame()
    
    def get_task_information(self, task_name: str) -> Optional[Dict]:
        """Get task information for a specific task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Optional[Dict]: Dictionary with task information or None if not found
        """
        # Look for task JSON file
        task_json_files = list(self.dataset_dir.rglob(f"*task-{task_name}*_bold.json"))
        
        if not task_json_files:
            logger.warning(f"No task information file found for task {task_name}")
            return None
        
        try:
            # Use the first task JSON file found
            task_info = load_json_file(task_json_files[0])
            logger.info(f"Loaded task information from {task_json_files[0]}")
            return task_info
            
        except Exception as e:
            logger.error(f"Failed to load task information: {str(e)}")
            return None 