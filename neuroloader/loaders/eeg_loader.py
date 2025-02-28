"""EEG dataset handling module"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import mne
from mne.io import read_raw_eeglab, read_raw_brainvision
from urllib.parse import urljoin
import numpy as np

from .base_loader import BaseDataset
from ..utils import find_files_by_extension, load_json_file, parse_bids_filename
from .. import logger

# Use the package's centralized logger
eeg_logger = logger.get_logger('loaders.eeg')

class EEGDataset(BaseDataset):
    """Class for handling EEG datasets from OpenNeuro.
    
    This class provides methods for downloading, processing, and analyzing EEG data
    in various formats (EEGLab, BrainVision, etc.)
    """
    
    def __init__(
        self, 
        dataset_id: str,
        data_dir: Optional[Union[str, Path]] = None,
        version: str = "latest"
    ):
        """Initialize an EEG dataset.
        
        Args:
            dataset_id: The unique identifier for the dataset on OpenNeuro
            data_dir: Directory where data will be stored (default: ./data)
            version: Dataset version to use (default: "latest")
        """
        super().__init__(dataset_id, data_dir, version)
        
        # EEG file extensions to search for
        self.eeg_extensions = [
            '.set', '.edf', '.bdf', '.vhdr', '.cnt', '.eeg'
        ]
        
        # Electrode files to look for
        self.electrode_file_patterns = [
            '*electrodes.tsv', '*channels.tsv', '*elec.txt'
        ]
        
        # Event file patterns
        self.event_file_patterns = [
            '*events.tsv', '*_events.json'
        ]
    
    def describe(self) -> Dict[str, Any]:
        """Get a detailed description of the EEG dataset.
        
        This method extends the base describe method with EEG-specific information.
        It provides a summary of recording files, formats, channels, and events.
        
        Returns:
            Dict[str, Any]: Dictionary containing dataset metadata and EEG-specific information
        """
        # Get base description (including subject information)
        description = super().describe()
        
        # Check if dataset is downloaded
        if not self.is_downloaded():
            return description
            
        # Get recording files and count by format
        recording_files = self.get_recording_files()
        
        # Count recordings by format
        format_counts = {}
        for file in recording_files:
            ext = file.suffix
            format_counts[ext] = format_counts.get(ext, 0) + 1
        
        # Get subject information from base class
        subject_ids = description.get('subjects', [])
        
        # Sample recording information (using the first file if available)
        sample_recording_info = {}
        if recording_files:
            try:
                # Load the first recording to get basic info without loading data
                first_recording = self.load_recording(recording_files[0], preload=False)
                if first_recording:
                    # Get channel types and count them
                    channel_types = first_recording.get_channel_types()
                    channel_type_counts = {}
                    for ch_type in channel_types:
                        channel_type_counts[ch_type] = channel_type_counts.get(ch_type, 0) + 1
                    
                    sample_recording_info = {
                        "sampling_rate": first_recording.info['sfreq'],
                        "channel_count": len(first_recording.ch_names),
                        "channel_types": channel_type_counts,  # Use the counts dictionary instead of direct conversion
                        "recording_length": first_recording.times[-1] if len(first_recording.times) > 0 else 0,
                        "montage": "Has electrode positions" if any(ch.get('loc', None) is not None and not all(v == 0 for v in ch['loc'][:3]) 
                                                              for ch in first_recording.info['chs']) else "No electrode positions"
                    }
                    
                    # Get events information if available
                    try:
                        events_df = self.get_events_dataframe(recording_files[0])
                        if not events_df.empty:
                            event_types = events_df['trial_type'].unique().tolist() if 'trial_type' in events_df.columns else []
                            sample_recording_info["event_types"] = event_types
                            sample_recording_info["event_count"] = len(events_df)
                    except Exception as e:
                        eeg_logger.warning(f"Could not extract events information: {str(e)}")
            except Exception as e:
                eeg_logger.warning(f"Could not load sample recording: {str(e)}")
        
        # Add EEG-specific information to description
        eeg_info = {
            "recording_count": len(recording_files),
            "recording_formats": format_counts,
            "sample_recording_info": sample_recording_info
        }
        
        # Add subject-specific recording counts if we have subjects
        if subject_ids:
            # Get recordings per subject using the base class's get_subject_files method
            recordings_per_subject = {}
            for subject_id in subject_ids:
                subject_files = self.get_subject_files(subject_id)
                eeg_files = [f for f in subject_files if f.suffix in self.eeg_extensions]
                recordings_per_subject[subject_id] = len(eeg_files)
            
            eeg_info["recordings_per_subject"] = recordings_per_subject
        
        # Merge with base description
        description["eeg_info"] = eeg_info
        
        return description
    
    def get_recording_files(self) -> List[Path]:
        """Get a list of EEG recording files in the dataset.
        
        Returns:
            List[Path]: List of paths to EEG files
        """
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            eeg_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return []
            
        return find_files_by_extension(self.dataset_dir, self.eeg_extensions)
    
    def get_subject_files(self, subject_id: str) -> List[Path]:
        """Get all files for a specific subject, with emphasis on EEG files.
        
        Overrides the base class method to use EEG-specific extensions.
        
        Args:
            subject_id: Subject ID to get files for
            
        Returns:
            List[Path]: List of paths to files for the subject
        """
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            eeg_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return []
            
        # Combine EEG extensions with common metadata extensions
        file_extensions = self.eeg_extensions + ['.json', '.tsv', '.txt']
        
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
                
            # Parse BIDS filename
            parts = parse_bids_filename(file_path.name)
            if parts.get('sub') == subject_id:
                subject_files.append(file_path)
        
        return subject_files
    
    def get_electrode_locations(self, recording_file: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Get electrode locations for a recording file.
        
        Two methods are used:
        1. Try to find a BIDS electrodes/channels file corresponding to the recording (preferred)
        2. Look for any electrode files in the same directory (fallback)
        
        Args:
            recording_file (Union[str, Path]): Path to the recording file
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with electrode locations or None if not found
        """
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            eeg_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return None
            
        recording_file = Path(recording_file)
        
        if not recording_file.exists():
            eeg_logger.error(f"Recording file does not exist: {recording_file}")
            return None
            
        # Use the filename directly for BIDS parsing
        recording_file_name = recording_file.name
        eeg_logger.info(f"Using filename for BIDS parsing: {recording_file_name}")
        
        # Method 1: Try BIDS naming convention first (primary method)
        try:
            # Parse the recording filename to extract BIDS components
            parts = parse_bids_filename(recording_file_name)
            eeg_logger.info(f"Parsed BIDS components: {parts}")
            
            if 'sub' in parts:
                # Build the expected electrode filename
                electrode_file_name = f"sub-{parts['sub']}"
                
                if 'ses' in parts:
                    electrode_file_name += f"_ses-{parts['ses']}"
                    
                if 'task' in parts:
                    electrode_file_name += f"_task-{parts['task']}"
                    
                if 'run' in parts:
                    electrode_file_name += f"_run-{parts['run']}"
                
                # Try both electrodes.tsv and channels.tsv
                for suffix in ['_electrodes.tsv', '_channels.tsv']:
                    electrode_file = recording_file.parent / (electrode_file_name + suffix)
                    eeg_logger.info(f"Looking for BIDS file: {electrode_file}")
                    
                    if electrode_file.exists():
                        try:
                            df = pd.read_csv(electrode_file, sep='\t')
                            eeg_logger.info(f"Loaded electrode data from BIDS-formatted file {electrode_file}")
                            return df
                        except Exception as e:
                            eeg_logger.warning(f"Failed to load BIDS file {electrode_file}: {str(e)}")
        except Exception as e:
            eeg_logger.warning(f"Failed to parse BIDS filename: {str(e)}")
        
        # Method 2: Simple fallback - look for any electrode files in the same directory
        recording_dir = recording_file.parent
        electrode_files = []
        
        # Use patterns to find any potential electrode files
        for pattern in self.electrode_file_patterns:
            electrode_files.extend(list(recording_dir.glob(pattern)))
        
        # If electrode files found, try to load the first one
        if electrode_files:
            electrode_file = electrode_files[0]
            try:
                if electrode_file.suffix == '.tsv':
                    df = pd.read_csv(electrode_file, sep='\t')
                    eeg_logger.info(f"Loaded electrodes from {electrode_file}")
                    return df
                elif electrode_file.suffix == '.txt':
                    df = pd.read_csv(electrode_file, sep='\s+')
                    eeg_logger.info(f"Loaded electrodes from {electrode_file}")
                    return df
                else:
                    eeg_logger.warning(f"Unsupported electrode file format: {electrode_file}")
            except Exception as e:
                eeg_logger.error(f"Failed to load electrode file: {str(e)}")
        
        # If no electrode file found
        eeg_logger.warning(f"No electrode location file found for {recording_file}")
        return None
    
    def get_events_dataframe(self, recording_file: Union[str, Path]) -> pd.DataFrame:
        """
        Get events information for a recording as a DataFrame.
        
        Two methods are used:
        1. Try to find a BIDS events file corresponding to the recording (preferred)
        2. Extract events directly from the recording using MNE (fallback)
        
        Args:
            recording_file (Union[str, Path]): Path to the recording file
            
        Returns:
            pd.DataFrame: DataFrame with events information
        """
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            eeg_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return pd.DataFrame()
            
        recording_file = Path(recording_file)
        
        if not recording_file.exists():
            eeg_logger.error(f"Recording file does not exist: {recording_file}")
            return pd.DataFrame()
        
        # Use the filename directly for BIDS parsing
        recording_file_name = recording_file.name
        eeg_logger.info(f"Using filename for BIDS parsing: {recording_file_name}")
        
        # Method 1: Try BIDS naming convention first
        try:
            # Parse the recording filename to extract BIDS components
            parts = parse_bids_filename(recording_file_name)
            eeg_logger.info(f"Parsed BIDS components: {parts}")
            
            if 'sub' in parts:
                # Build the expected events filename based on BIDS naming conventions
                events_file_name = f"sub-{parts['sub']}"
                
                if 'ses' in parts:
                    events_file_name += f"_ses-{parts['ses']}"
                    
                if 'task' in parts:
                    events_file_name += f"_task-{parts['task']}"
                    
                if 'run' in parts:
                    events_file_name += f"_run-{parts['run']}"
                
                # Add events suffix
                events_file_name += "_events.tsv"
                
                # Look for this file in the recording directory
                events_file = recording_file.parent / events_file_name
                eeg_logger.info(f"Looking for BIDS events file: {events_file}")
                
                if events_file.exists():
                    try:
                        events_df = pd.read_csv(events_file, sep='\t')
                        eeg_logger.info(f"Loaded events from BIDS-formatted file {events_file}")
                        return events_df
                    except Exception as e:
                        eeg_logger.warning(f"Failed to load BIDS events file {events_file}: {str(e)}")
        except Exception as e:
            eeg_logger.warning(f"Failed to parse BIDS filename: {str(e)}")
        
        # Method 2: Simple fallback - look for any events files in the same directory
        recording_dir = recording_file.parent
        events_files = []
        
        # Use more generic patterns to find any potential events files
        for pattern in ['*_events.tsv', '*events*.tsv', '*_events.json', '*events*.json']:
            events_files.extend(list(recording_dir.glob(pattern)))
        
        # If events files found, try to load the first one
        if events_files:
            events_file = events_files[0]
            try:
                if events_file.suffix == '.tsv':
                    events_df = pd.read_csv(events_file, sep='\t')
                    eeg_logger.info(f"Loaded events from {events_file}")
                    return events_df
                elif events_file.suffix == '.json':
                    events_dict = load_json_file(events_file)
                    # Convert JSON to DataFrame
                    events_df = pd.DataFrame(events_dict)
                    eeg_logger.info(f"Loaded events from {events_file}")
                    return events_df
            except Exception as e:
                eeg_logger.warning(f"Failed to load events file {events_file}: {str(e)}")
        
        # Method 3: If no events file found, extract from recording using MNE
        try:
            eeg_logger.info(f"Extracting events from recording file {recording_file}")
            
            # Determine file format and load with appropriate MNE function
            if recording_file.suffix == '.set':
                raw = read_raw_eeglab(recording_file, preload=False)
            elif recording_file.suffix == '.vhdr':
                raw = read_raw_brainvision(recording_file, preload=False)
            elif recording_file.suffix in ['.edf', '.bdf']:
                raw = mne.io.read_raw(recording_file, preload=False)
            else:
                eeg_logger.error(f"Unsupported file format: {recording_file.suffix}")
                return pd.DataFrame()
            
            # Try to extract events from annotations first
            try:
                eeg_logger.info("Trying to extract events from annotations...")
                events, event_id = mne.events_from_annotations(raw)
                
                if len(events) > 0:
                    # Create DataFrame
                    events_df = pd.DataFrame({
                        'onset': events[:, 0] / raw.info['sfreq'],  # Convert to seconds
                        'duration': 0.0,  # Default duration
                        'trial_type': [list(event_id.keys())[list(event_id.values()).index(e)] 
                                      if e in list(event_id.values()) else str(e) 
                                      for e in events[:, 2]]
                    })
                    
                    eeg_logger.info(f"Extracted {len(events_df)} events from recording annotations")
                    return events_df
                else:
                    eeg_logger.warning("No events found in annotations")
            except Exception as e:
                eeg_logger.warning(f"Failed to extract events from annotations: {str(e)}")
            
            # If annotations didn't work, try MNE's find_events
            try:
                eeg_logger.info("Trying to extract events using MNE's find_events...")
                # Try with auto-detection of stim channels
                events = mne.find_events(raw, stim_channel=None, consecutive=True)
                
                if len(events) > 0:
                    # Create DataFrame
                    events_df = pd.DataFrame({
                        'onset': events[:, 0] / raw.info['sfreq'],  # Convert to seconds
                        'duration': 0.0,  # Default duration
                        'trial_type': events[:, 2].astype(str)  # Use event codes as trial types
                    })
                    eeg_logger.info(f"Extracted {len(events_df)} events using MNE's find_events")
                    return events_df
            except Exception as e:
                eeg_logger.warning(f"Failed to find events: {str(e)}")
            
            # If we still don't have events, return empty DataFrame
            eeg_logger.warning("No events found in recording using any method")
            return pd.DataFrame()
        
        except Exception as e:
            eeg_logger.error(f"Failed to extract events from recording: {str(e)}")
            return pd.DataFrame()
    
    def load_recording(self, recording_file: Union[str, Path], preload: bool = False) -> Optional[mne.io.Raw]:
        """Load an EEG recording file with MNE.
        
        Args:
            recording_file: Path to the recording file
            preload: Whether to preload data into memory
            
        Returns:
            Optional[mne.io.Raw]: MNE Raw object or None if loading fails
        """
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            eeg_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return None
            
        recording_file = Path(recording_file)
        
        if not recording_file.exists():
            eeg_logger.error(f"Recording file does not exist: {recording_file}")
            return None
        
        try:
            eeg_logger.info(f"Loading recording from {recording_file}")
            
            # Determine file format and load with appropriate MNE function
            if recording_file.suffix == '.set':
                raw = read_raw_eeglab(recording_file, preload=preload)
            elif recording_file.suffix == '.vhdr':
                raw = read_raw_brainvision(recording_file, preload=preload)
            elif recording_file.suffix in ['.edf', '.bdf']:
                raw = mne.io.read_raw(recording_file, preload=preload)
            else:
                eeg_logger.error(f"Unsupported file format: {recording_file.suffix}")
                return None
            
            eeg_logger.info(f"Successfully loaded recording with {len(raw.ch_names)} channels")
            return raw
            
        except Exception as e:
            eeg_logger.error(f"Failed to load recording: {str(e)}")
            return None 