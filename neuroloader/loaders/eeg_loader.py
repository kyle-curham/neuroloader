"""EEG dataset handling module"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import logging
import mne
from mne.io import read_raw_eeglab, read_raw_brainvision
from urllib.parse import urljoin

from .base_loader import BaseDataset
from ..utils import find_files_by_extension, load_json_file, parse_bids_filename

logger = logging.getLogger(__name__)

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
    
    def get_recording_files(self) -> List[Path]:
        """Get a list of EEG recording files in the dataset.
        
        Returns:
            List[Path]: List of paths to EEG files
        """
        return find_files_by_extension(self.dataset_dir, self.eeg_extensions)
    
    def get_electrode_locations(self, recording_file: Union[str, Path]) -> Optional[pd.DataFrame]:
        """Get electrode locations for a recording file.
        
        Args:
            recording_file: Path to the recording file
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with electrode locations or None if not found
        """
        recording_file = Path(recording_file)
        
        if not recording_file.exists():
            logger.error(f"Recording file does not exist: {recording_file}")
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
            logger.warning(f"No electrode location file found for {recording_file}")
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
                logger.warning(f"Unsupported electrode file format: {electrode_file}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load electrode file: {str(e)}")
            return None
    
    def get_events_dataframe(self, recording_file: Union[str, Path]) -> pd.DataFrame:
        """Create a DataFrame containing events for the specified recording.
        
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
                    logger.info(f"Loaded events from {events_file}")
                    return events_df
                elif events_file.suffix == '.json':
                    events_dict = load_json_file(events_file)
                    # Convert JSON to DataFrame (format depends on JSON structure)
                    events_df = pd.DataFrame(events_dict)
                    logger.info(f"Loaded events from {events_file}")
                    return events_df
            except Exception as e:
                logger.warning(f"Failed to load events file {events_file}: {str(e)}")
                # Fall back to extracting from the recording file
        
        # If no events file found or loading failed, try to extract from recording
        try:
            logger.info(f"Extracting events from recording file {recording_file}")
            
            # Determine file format and load with appropriate MNE function
            if recording_file.suffix == '.set':
                raw = read_raw_eeglab(recording_file, preload=False)
            elif recording_file.suffix == '.vhdr':
                raw = read_raw_brainvision(recording_file, preload=False)
            elif recording_file.suffix in ['.edf', '.bdf']:
                raw = mne.io.read_raw(recording_file, preload=False)
            else:
                logger.error(f"Unsupported file format: {recording_file.suffix}")
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
            
            logger.info(f"Extracted {len(events_df)} events from recording")
            return events_df
            
        except Exception as e:
            logger.error(f"Failed to extract events from recording: {str(e)}")
            return pd.DataFrame()
    
    def load_recording(self, recording_file: Union[str, Path], preload: bool = False) -> Optional[mne.io.Raw]:
        """Load an EEG recording file with MNE.
        
        Args:
            recording_file: Path to the recording file
            preload: Whether to preload data into memory
            
        Returns:
            Optional[mne.io.Raw]: MNE Raw object or None if loading fails
        """
        recording_file = Path(recording_file)
        
        if not recording_file.exists():
            logger.error(f"Recording file does not exist: {recording_file}")
            return None
        
        try:
            logger.info(f"Loading recording from {recording_file}")
            
            # Determine file format and load with appropriate MNE function
            if recording_file.suffix == '.set':
                raw = read_raw_eeglab(recording_file, preload=preload)
            elif recording_file.suffix == '.vhdr':
                raw = read_raw_brainvision(recording_file, preload=preload)
            elif recording_file.suffix in ['.edf', '.bdf']:
                raw = mne.io.read_raw(recording_file, preload=preload)
            else:
                logger.error(f"Unsupported file format: {recording_file.suffix}")
                return None
            
            logger.info(f"Successfully loaded recording with {len(raw.ch_names)} channels")
            return raw
            
        except Exception as e:
            logger.error(f"Failed to load recording: {str(e)}")
            return None 