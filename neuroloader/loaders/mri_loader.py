"""MRI dataset handling module"""

import os
import json
import requests
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from urllib.parse import urljoin

from .base_loader import BaseDataset
from ..utils import find_files_by_extension, load_json_file, parse_bids_filename
from .. import logger

# Use the package's centralized logger
mri_logger = logger.get_logger('loaders.mri')

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
    
    def describe(self) -> Dict[str, Any]:
        """Get a detailed description of the MRI dataset.
        
        This method extends the base describe method with MRI-specific information.
        It provides a summary of scan types, formats, and dimensions.
        
        Returns:
            Dict[str, Any]: Dictionary containing dataset metadata and MRI-specific information
        """
        # Get base description
        description = super().describe()
        
        # Check if dataset is downloaded
        if not self.is_downloaded():
            description["download_status"] = "Not downloaded"
            return description
            
        # Get recording files
        recording_files = self.get_recording_files()
        
        # Count by file format
        format_counts = {}
        for file in recording_files:
            ext = file.suffix
            format_counts[ext] = format_counts.get(ext, 0) + 1
        
        # Count by scan type based on filename patterns
        scan_type_counts = {}
        for file in recording_files:
            # Parse filename to extract BIDS components
            components = parse_bids_filename(file.name)
            if components and 'modality' in components:
                scan_type = components['modality']
                scan_type_counts[scan_type] = scan_type_counts.get(scan_type, 0) + 1
        
        # Sample scan information (using the first file if available)
        sample_scan_info = {}
        if recording_files:
            try:
                # Load the first scan to get basic info
                scan_img, scan_metadata = self.load_scan(recording_files[0])
                if scan_img:
                    sample_scan_info = {
                        "dimensions": scan_img.shape,
                        "voxel_sizes": scan_img.header.get_zooms(),
                        "data_type": str(scan_img.get_data_dtype()),
                    }
                    
                if scan_metadata:
                    # Include relevant metadata fields
                    important_metadata = {k: v for k, v in scan_metadata.items() 
                                         if k in ('RepetitionTime', 'EchoTime', 'Manufacturer', 
                                                  'MagneticFieldStrength', 'TaskName')}
                    sample_scan_info["metadata"] = important_metadata
            except Exception as e:
                mri_logger.warning(f"Could not load sample scan: {str(e)}")
        
        # Add MRI-specific information to description
        mri_info = {
            "scan_count": len(recording_files),
            "file_formats": format_counts,
            "scan_types": scan_type_counts,
            "sample_scan_info": sample_scan_info
        }
        
        # Merge with base description
        description["mri_info"] = mri_info
        description["download_status"] = "Downloaded"
        
        return description
    
    def get_recording_files(self) -> List[Path]:
        """Get a list of MRI recording files in the dataset.
        
        Returns:
            List[Path]: List of paths to MRI files
        """
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            mri_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return []
            
        return find_files_by_extension(self.dataset_dir, self.mri_extensions)
    
    def get_structural_scans(self, scan_type: str = "T1w") -> List[Path]:
        """Get all structural MRI scans of a specific type.
        
        Args:
            scan_type: Type of structural scan to find (e.g., "T1w", "T2w")
            
        Returns:
            List[Path]: List of paths to structural scan files
        """
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            mri_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return []
            
        all_mri_files = self.get_recording_files()
        
        # Filter by scan type
        structural_scans = []
        for file_path in all_mri_files:
            if scan_type.lower() in file_path.name.lower():
                structural_scans.append(file_path)
        
        mri_logger.info(f"Found {len(structural_scans)} {scan_type} structural scans")
        return structural_scans
    
    def load_scan(self, scan_file: Union[str, Path]) -> Tuple[Optional[nib.Nifti1Image], Optional[Dict]]:
        """Load an MRI scan file with nibabel and its metadata if available.
        
        Args:
            scan_file: Path to the scan file
            
        Returns:
            Tuple[Optional[nib.Nifti1Image], Optional[Dict]]: 
                Tuple of (NIfTI image object, metadata dictionary)
        """
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            mri_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return None, None
            
        scan_file = Path(scan_file)
        
        if not scan_file.exists():
            mri_logger.error(f"Scan file does not exist: {scan_file}")
            return None, None
        
        try:
            # Load the NIfTI file
            mri_logger.info(f"Loading scan from {scan_file}")
            img = nib.load(scan_file)
            
            # Try to find metadata file
            metadata = None
            metadata_path = scan_file.with_suffix('.json')
            
            if metadata_path.exists():
                try:
                    metadata = load_json_file(metadata_path)
                    mri_logger.info(f"Loaded metadata from {metadata_path}")
                except Exception as e:
                    mri_logger.warning(f"Failed to load metadata: {str(e)}")
            
            return img, metadata
            
        except Exception as e:
            mri_logger.error(f"Failed to load scan: {str(e)}")
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
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            mri_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return pd.DataFrame()
            
        recording_file = Path(recording_file)
        
        if not recording_file.exists():
            mri_logger.error(f"Recording file does not exist: {recording_file}")
            return pd.DataFrame()
        
        # Try to find events file in the same directory
        events_file = recording_file.with_name(recording_file.stem + '_events.tsv')
        
        if not events_file.exists():
            # Try to find other events files in the same directory
            events_files = list(recording_file.parent.glob('*_events.tsv'))
            if events_files:
                events_file = events_files[0]
            else:
                mri_logger.warning(f"No events file found for {recording_file}")
                return pd.DataFrame()
        
        try:
            # Load events from TSV file
            events_df = pd.read_csv(events_file, sep='\t')
            mri_logger.info(f"Loaded events from {events_file}")
            return events_df
            
        except Exception as e:
            mri_logger.error(f"Failed to load events file: {str(e)}")
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
    
    def describe(self) -> Dict[str, Any]:
        """Get a detailed description of the fMRI dataset.
        
        This method extends the MRIDataset describe method with fMRI-specific information.
        It provides a summary of functional scan types, task information, and events.
        
        Returns:
            Dict[str, Any]: Dictionary containing dataset metadata and fMRI-specific information
        """
        # Get base description from MRIDataset
        description = super().describe()
        
        # Check if dataset is downloaded
        if not self.is_downloaded():
            return description
            
        # Get functional scans
        functional_scans = self.get_functional_scans()
        
        # Count by task
        task_counts = {}
        for file in functional_scans:
            # Parse filename to extract BIDS components
            components = parse_bids_filename(file.name)
            if components and 'task' in components:
                task = components['task']
                task_counts[task] = task_counts.get(task, 0) + 1
        
        # Get available task information
        tasks_info = {}
        for task in task_counts.keys():
            try:
                task_metadata = self.get_task_information(task)
                if task_metadata:
                    # Extract key information
                    task_info = {
                        "description": task_metadata.get("TaskDescription", ""),
                        "instructions": task_metadata.get("Instructions", ""),
                        "cognitive_paradigm": task_metadata.get("CognitiveParadigm", ""),
                        "response_options": task_metadata.get("ResponseOptions", [])
                    }
                    tasks_info[task] = task_info
            except Exception as e:
                mri_logger.warning(f"Failed to get task information for {task}: {str(e)}")
        
        # Sample events information
        sample_events_info = {}
        if functional_scans:
            try:
                # Get events from first functional scan
                events_df = self.get_events_dataframe(functional_scans[0])
                if not events_df.empty:
                    # Get event types and counts
                    if 'trial_type' in events_df.columns:
                        event_types = events_df['trial_type'].value_counts().to_dict()
                        sample_events_info["event_types"] = event_types
                    else:
                        # Use unique combinations of available columns as event types
                        sample_events_info["columns"] = events_df.columns.tolist()
                    
                    sample_events_info["total_events"] = len(events_df)
                    sample_events_info["duration_range"] = [
                        events_df['duration'].min() if 'duration' in events_df.columns else 0,
                        events_df['duration'].max() if 'duration' in events_df.columns else 0
                    ]
            except Exception as e:
                mri_logger.warning(f"Failed to extract sample events information: {str(e)}")
        
        # Add fMRI-specific information
        fmri_info = description.get("mri_info", {})
        fmri_info.update({
            "functional_scan_count": len(functional_scans),
            "tasks": task_counts,
            "task_details": tasks_info,
            "sample_events_info": sample_events_info,
            "is_functional": True
        })
        
        # Update the mri_info with fMRI details
        description["mri_info"] = fmri_info
        
        return description
    
    def get_functional_scans(self) -> List[Path]:
        """Get all functional MRI scans in the dataset.
        
        Returns:
            List[Path]: List of paths to functional scan files
        """
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            mri_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return []
            
        all_mri_files = self.get_recording_files()
        
        # Filter for functional scans (containing 'bold', 'func', or 'task')
        functional_scans = []
        for file_path in all_mri_files:
            if any(keyword in file_path.name.lower() for keyword in ['bold', 'func', 'task']):
                functional_scans.append(file_path)
        
        mri_logger.info(f"Found {len(functional_scans)} functional scans")
        return functional_scans
    
    def get_events_dataframe(self, recording_file: Union[str, Path]) -> pd.DataFrame:
        """Create a DataFrame containing events for the specified fMRI recording.
        
        This method overrides the base implementation with fMRI-specific event handling.
        
        Args:
            recording_file: Path to the recording file
            
        Returns:
            pd.DataFrame: DataFrame with events information
        """
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            mri_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return pd.DataFrame()
            
        recording_file = Path(recording_file)
        
        if not recording_file.exists():
            mri_logger.error(f"Recording file does not exist: {recording_file}")
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
                    mri_logger.warning(f"No events file found for {recording_file}")
                    return pd.DataFrame()
        else:
            # Fallback to simple name-based matching
            events_file = recording_file.with_name(recording_file.stem + '_events.tsv')
            
            if not events_file.exists():
                events_files = list(recording_file.parent.glob('*_events.tsv'))
                if events_files:
                    events_file = events_files[0]
                else:
                    mri_logger.warning(f"No events file found for {recording_file}")
                    return pd.DataFrame()
        
        try:
            # Load events from TSV file
            events_df = pd.read_csv(events_file, sep='\t')
            
            # Ensure required columns exist
            if 'onset' not in events_df.columns:
                mri_logger.warning(f"Events file {events_file} missing required 'onset' column")
                return pd.DataFrame()
                
            mri_logger.info(f"Loaded events from {events_file} with {len(events_df)} events")
            return events_df
            
        except Exception as e:
            mri_logger.error(f"Failed to load events file: {str(e)}")
            return pd.DataFrame()
            
    def get_task_information(self, task_name: str) -> Optional[Dict]:
        """Get information about a specific task from the dataset.
        
        Args:
            task_name: Name of the task to get information for
            
        Returns:
            Optional[Dict]: Dictionary with task information or None if not found
        """
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            mri_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return None
            
        # Try to find task JSON file (*_task-<task_name>_*.json)
        task_files = list(self.dataset_dir.rglob(f"*task-{task_name}*.json"))
        
        if not task_files:
            mri_logger.warning(f"No task information found for task '{task_name}'")
            return None
            
        # Use the first found task file
        task_file = task_files[0]
        
        try:
            task_info = load_json_file(task_file)
            mri_logger.info(f"Loaded task information from {task_file}")
            return task_info
            
        except Exception as e:
            mri_logger.error(f"Failed to load task information: {str(e)}")
            return None 