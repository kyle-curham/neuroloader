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
        # Get base description
        description = super().describe()
        
        # Check if dataset is downloaded
        if not self.is_downloaded():
            description["download_status"] = "Not downloaded"
            return description
            
        # Get recording files and count by format
        recording_files = self.get_recording_files()
        
        # Count recordings by format
        format_counts = {}
        for file in recording_files:
            ext = file.suffix
            format_counts[ext] = format_counts.get(ext, 0) + 1
        
        # Sample recording information (using the first file if available)
        sample_recording_info = {}
        if recording_files:
            try:
                # Load the first recording to get basic info without loading data
                first_recording = self.load_recording(recording_files[0], preload=False)
                if first_recording:
                    sample_recording_info = {
                        "sampling_rate": first_recording.info['sfreq'],
                        "channel_count": len(first_recording.ch_names),
                        "channel_types": dict(first_recording.get_channel_types()),
                        "recording_length": first_recording.times[-1] if len(first_recording.times) > 0 else 0,
                        "montage": str(first_recording.get_montage()) if hasattr(first_recording, 'get_montage') else None
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
        
        # Merge with base description
        description["eeg_info"] = eeg_info
        description["download_status"] = "Downloaded"
        
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
    
    def get_electrode_locations(self, recording_file: Union[str, Path]) -> Optional[pd.DataFrame]:
        """Get electrode locations for a recording file.
        
        Args:
            recording_file: Path to the recording file
            
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
        
        # Try to find electrode file in the same directory
        recording_dir = recording_file.parent
        electrode_files = []
        
        for pattern in self.electrode_file_patterns:
            electrode_files.extend(list(recording_dir.glob(pattern)))
        
        if not electrode_files:
            # Try to find in the parent directory
            electrode_files = []
            for pattern in self.electrode_file_patterns:
                electrode_files.extend(list(recording_dir.parent.glob(pattern)))
        
        if not electrode_files:
            eeg_logger.warning(f"No electrode location file found for {recording_file}")
            return None
        
        # Use the first found electrode file
        electrode_file = electrode_files[0]
        
        try:
            # Load based on file extension
            if electrode_file.suffix == '.tsv':
                return pd.read_csv(electrode_file, sep='\t')
            elif electrode_file.suffix == '.txt':
                return pd.read_csv(electrode_file, sep='\s+')
            else:
                eeg_logger.warning(f"Unsupported electrode file format: {electrode_file}")
                return None
                
        except Exception as e:
            eeg_logger.error(f"Failed to load electrode file: {str(e)}")
            return None
    
    def get_events_dataframe(self, recording_file: Union[str, Path]) -> pd.DataFrame:
        """Create a DataFrame containing events for the specified recording.
        
        Args:
            recording_file: Path to the recording file
            
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
        
        # Try to find events file in the same directory
        recording_dir = recording_file.parent
        events_files = []
        
        for pattern in self.event_file_patterns:
            events_files.extend(list(recording_dir.glob(pattern)))
        
        # If found, load from events file
        if events_files:
            events_file = events_files[0]
            try:
                if events_file.suffix == '.tsv':
                    events_df = pd.read_csv(events_file, sep='\t')
                    eeg_logger.info(f"Loaded events from {events_file}")
                    return events_df
                elif events_file.suffix == '.json':
                    events_dict = load_json_file(events_file)
                    # Convert JSON to DataFrame (format depends on JSON structure)
                    events_df = pd.DataFrame(events_dict)
                    eeg_logger.info(f"Loaded events from {events_file}")
                    return events_df
            except Exception as e:
                eeg_logger.warning(f"Failed to load events file {events_file}: {str(e)}")
                # Fall back to extracting from the recording file
        
        # If no events file found or loading failed, try to extract from recording
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
            
            # Extract events
            events, event_id = mne.events_from_annotations(raw)
            
            # Create DataFrame
            events_df = pd.DataFrame({
                'onset': events[:, 0] / raw.info['sfreq'],  # Convert to seconds
                'duration': 0.0,  # Default duration
                'trial_type': [list(event_id.keys())[list(event_id.values()).index(e)] 
                              if e in list(event_id.values()) else str(e) 
                              for e in events[:, 2]]
            })
            
            eeg_logger.info(f"Extracted {len(events_df)} events from recording")
            return events_df
            
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