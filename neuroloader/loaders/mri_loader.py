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
from ..utils import find_files_by_extension, load_json_file, parse_bids_filename, build_bids_filename
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
    
    def get_recording_files(self, subject_id: str = None, acquisition: str = None,
                           task: str = None, run: str = None, return_paths: bool = True,
                           file_type: str = 'nii.gz', include_fieldmaps: bool = False) -> List[Path]:
        """Get a list of recording files for the given subject and optional filters."""
        if subject_id and not self._is_valid_subject_id(subject_id):
            mri_logger.warning(f"Invalid subject ID format: {subject_id}")
            return []
            
        # Get path to subject directory
        subject_dir = None
        if subject_id:
            subject_dir = self.dataset_dir / f"sub-{subject_id}"
            if not subject_dir.exists():
                mri_logger.error(f"Subject directory does not exist: {subject_dir}")
                return []
                
        # Find MRI files
        mri_dirs = []
        if subject_dir:
            # Look in the subject's MRI directory
            sub_mri_dir = subject_dir / "anat"
            sub_func_dir = subject_dir / "func"
            sub_dwi_dir = subject_dir / "dwi"
            if sub_mri_dir.exists():
                mri_dirs.append(sub_mri_dir)
            if sub_func_dir.exists():
                mri_dirs.append(sub_func_dir)
            if sub_dwi_dir.exists():
                mri_dirs.append(sub_dwi_dir)
            if include_fieldmaps:
                sub_fmap_dir = subject_dir / "fmap"
                if sub_fmap_dir.exists():
                    mri_dirs.append(sub_fmap_dir)
        else:
            # Look in all subject directories
            mri_logger.info(f"Looking for MRI recordings in all subjects")
            subject_dirs = [d for d in self.dataset_dir.glob("sub-*") if d.is_dir()]
            for sub_dir in subject_dirs:
                sub_mri_dir = sub_dir / "anat"
                sub_func_dir = sub_dir / "func"
                sub_dwi_dir = sub_dir / "dwi"
                if sub_mri_dir.exists():
                    mri_dirs.append(sub_mri_dir)
                if sub_func_dir.exists():
                    mri_dirs.append(sub_func_dir)
                if sub_dwi_dir.exists():
                    mri_dirs.append(sub_dwi_dir)
                if include_fieldmaps:
                    sub_fmap_dir = sub_dir / "fmap"
                    if sub_fmap_dir.exists():
                        mri_dirs.append(sub_fmap_dir)
                
        # Find all MRI files
        mri_files = []
        for mri_dir in mri_dirs:
            found_files = find_files_by_extension(mri_dir, file_type)
            mri_files.extend(found_files)
            
        # Filter files based on task and run if provided
        filtered_files = []
        for file_path in mri_files:
            # Use the filename directly for BIDS parsing
            file_path_name = file_path.name
            mri_logger.info(f"Checking file: {file_path_name}")
            
            parts = parse_bids_filename(file_path_name)
            
            if task and ('task' in parts and parts['task'] != task):
                continue
                
            if run and ('run' in parts and parts['run'] != run):
                continue
                
            if acquisition and ('acq' in parts and parts['acq'] != acquisition):
                continue
                
            filtered_files.append(file_path)
            
        return filtered_files
    
    def get_subject_files(self, subject_id: str) -> List[Path]:
        """Get all files for a specific subject, with emphasis on MRI files.
        
        Args:
            subject_id: Subject ID to get files for
            
        Returns:
            List[Path]: List of paths to files for the subject
        """
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            mri_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return []
            
        # Combine MRI extensions with common metadata extensions
        file_extensions = self.mri_extensions + ['.json', '.tsv', '.txt', '.bval', '.bvec']
        
        all_files = find_files_by_extension(self.dataset_dir, file_extensions)
        
        # Filter files by subject ID
        subject_files = []
        
        for file_path in all_files:
            # Check if subject ID is in the filename
            if f'sub-{subject_id}' in file_path.name:
                subject_files.append(file_path)
                continue
                
            # Check if subject ID is in the directory path
            if f'sub-{subject_id}' in str(file_path):
                subject_files.append(file_path)
                continue
                
            # Resolve potential DataLad/git-annex hashed filenames
            _, resolved_filename = self.resolve_real_filename(file_path)
            
            # Parse BIDS filename with the resolved name
            parts = parse_bids_filename(resolved_filename)
            if parts.get('sub') == subject_id:
                subject_files.append(file_path)
        
        return subject_files
    
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
            # Use the filename directly for BIDS parsing
            scan_file_name = scan_file.name
            mri_logger.info(f"Using filename for processing: {scan_file_name}")
            
            # Load the NIfTI file
            mri_logger.info(f"Loading scan from {scan_file}")
            img = nib.load(scan_file)
            
            # Try to find metadata file
            metadata = None
            parts = parse_bids_filename(scan_file_name)
            
            if 'modality' in parts:
                json_filename = scan_file.with_suffix('.json')
                if json_filename.exists():
                    try:
                        with open(json_filename, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        mri_logger.warning(f"Error loading metadata file {json_filename}: {e}")
                else:
                    mri_logger.info(f"No metadata file found for {scan_file}")
            
            return img, metadata
        except Exception as e:
            mri_logger.error(f"Error loading scan file {scan_file}: {e}")
            return None, None
    
    def get_events_dataframe(self, recording_file: Union[str, Path]) -> pd.DataFrame:
        """Load events data for a functional MRI recording."""
        if not recording_file:
            mri_logger.error("No recording file provided")
            return pd.DataFrame()
            
        recording_file = Path(recording_file)
        
        if not recording_file.exists():
            mri_logger.error(f"Recording file does not exist: {recording_file}")
            return pd.DataFrame()
        
        # Use the filename directly for BIDS parsing
        recording_file_name = recording_file.name
        mri_logger.info(f"Using filename for BIDS parsing: {recording_file_name}")
        
        # Method 1: Try BIDS naming convention (primary method)
        try:
            # Parse the recording filename to extract BIDS components
            parts = parse_bids_filename(recording_file_name)
            mri_logger.info(f"Parsed BIDS components: {parts}")
            
            # Check if this is a functional MRI file
            if 'modality' not in parts or parts['modality'] != 'bold':
                mri_logger.warning(f"Not a functional MRI file: {recording_file}")
                return pd.DataFrame()
                
            if 'subject' not in parts or 'task' not in parts:
                mri_logger.warning(f"Missing required BIDS components in filename: {recording_file}")
                return pd.DataFrame()
            
            # Construct events filename using BIDS convention
            events_file_parts = {
                'subject': parts['subject'],
                'task': parts['task'],
                'modality': 'events'
            }
            
            # Include run number if present in the original file
            if 'run' in parts:
                events_file_parts['run'] = parts['run']
                
            events_filename = build_bids_filename(events_file_parts) + '.tsv'
            events_file = recording_file.parent / events_filename
            mri_logger.info(f"Looking for BIDS events file: {events_file}")
            
            if events_file.exists():
                try:
                    events_df = pd.read_csv(events_file, sep='\t')
                    mri_logger.info(f"Loaded events from BIDS-formatted file: {events_file}")
                    return events_df
                except Exception as e:
                    mri_logger.warning(f"Failed to load BIDS events file {events_file}: {e}")
            else:
                mri_logger.warning(f"BIDS events file not found: {events_file}")
        except Exception as e:
            mri_logger.warning(f"Failed to parse or find BIDS events file: {e}")
            
        # Fallback Method: Look for any events files in the same directory
        mri_logger.info("Trying fallback methods to find events file")
        recording_dir = recording_file.parent
        
        # Look for events files in the recording directory
        events_patterns = ['*_events.tsv', '*events*.tsv']
        for pattern in events_patterns:
            event_files = list(recording_dir.glob(pattern))
            if event_files:
                try:
                    events_file = event_files[0]
                    mri_logger.info(f"Found fallback events file: {events_file}")
                    events_df = pd.read_csv(events_file, sep='\t')
                    mri_logger.info(f"Loaded events from fallback file: {events_file}")
                    return events_df
                except Exception as e:
                    mri_logger.warning(f"Failed to load fallback events file {events_file}: {e}")
                    
        # No events files found
        mri_logger.warning("No events file found for recording")
        return pd.DataFrame()


class FMRIDataset(MRIDataset):
    """Class for handling fMRI datasets from OpenNeuro.
    
    This class extends MRIDataset to focus on functional MRI data,
    adding functionality for task information, events, and BOLD signals.
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
        
        # Additional patterns specific to fMRI
        self.fmri_specific_patterns = [
            '*bold.nii.gz', '*bold.nii', '*_cbv.nii.gz', '*_cbf.nii.gz'
        ]
        
    def get_subject_files(self, subject_id: str) -> List[Path]:
        """Get all files for a specific subject, with emphasis on fMRI files.
        
        This overrides the MRIDataset implementation to include fMRI-specific files.
        
        Args:
            subject_id: Subject ID to get files for
            
        Returns:
            List[Path]: List of paths to files for the subject
        """
        # Get basic MRI files from parent class implementation
        subject_files = super().get_subject_files(subject_id)
        
        # Add fMRI-specific files
        # These would be covered by the parent class implementation since the file extensions
        # are the same (.nii, .nii.gz), but we'll search specifically for fMRI patterns
        # to ensure we don't miss any
        
        # First, find all *bold* files
        
        for pattern in self.fmri_specific_patterns:
            # Filter for this subject
            subject_pattern = f"sub-{subject_id}*{pattern}"
            fmri_files = list(self.dataset_dir.rglob(subject_pattern))
            
            # Add to subject files list
            for file_path in fmri_files:
                if file_path not in subject_files:
                    subject_files.append(file_path)
        
        # Also look for event files associated with this subject
        events_pattern = f"sub-{subject_id}*_events.tsv"
        event_files = list(self.dataset_dir.rglob(events_pattern))
        for file_path in event_files:
            if file_path not in subject_files:
                subject_files.append(file_path)
        
        return subject_files
        
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