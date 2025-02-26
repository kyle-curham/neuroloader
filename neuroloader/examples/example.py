"""
Example script for downloading and exploring a preprocessed EEG dataset using neuroloader.

This example demonstrates:
1. Initializing an EEG dataset
2. Downloading it using DataLad's Python API
3. Exploring the EEG data contents with specialized EEG methods
"""

from neuroloader import create_dataset, EEGDataset, logger
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import sys

def explore_eeg_dataset(dataset: EEGDataset) -> None:
    """
    Explore the contents of an EEG dataset, focusing on EEG-specific features.
    
    Args:
        dataset: The EEGDataset object to explore
    """
    # Get a component-specific logger
    eeg_logger = logger.get_logger('examples.explore_eeg')
    
    # Get dataset information
    logger.log_step(eeg_logger, "Exploring dataset information")
    dataset_info = dataset.describe()
    
    if hasattr(dataset_info, 'get'):
        subject_count = dataset_info.get('subject_count', 'unknown')
        subjects = dataset_info.get('subjects', [])
        eeg_logger.info(f"Dataset contains {subject_count} subjects")
        if subjects:
            eeg_logger.info(f"Subject IDs: {subjects[:5]}...")
    
    # List EEG recording files
    eeg_logger.info("Finding EEG recording files")
    recording_files = dataset.get_recording_files()
    eeg_logger.info(f"Found {len(recording_files)} EEG files")
    
    # Show first few recording files
    for i, file_path in enumerate(recording_files[:5]):
        eeg_logger.info(f"  {i+1}. {file_path}")
    
    # Check if there are any recordings
    if recording_files:
        # Get the first recording file for exploration
        example_file = recording_files[0]
        eeg_logger.info(f"Exploring example EEG file: {example_file.name}")
        
        # Get events for the example recording
        try:
            eeg_logger.info("Extracting events information")
            events_df = dataset.get_events_dataframe(example_file)
            eeg_logger.info(f"Found {len(events_df)} events")
            if not events_df.empty:
                eeg_logger.info(f"Events sample:\n{events_df.head()}")
                
                # Count occurrences of different event types
                if 'trial_type' in events_df.columns:
                    event_counts = events_df['trial_type'].value_counts()
                    eeg_logger.info("Event Type Counts:")
                    for event_type, count in event_counts.items():
                        eeg_logger.info(f"  {event_type}: {count}")
        except Exception as e:
            logger.log_exception(eeg_logger, e, "Error reading events")
        
        # Try to get electrode locations
        try:
            eeg_logger.info("Getting electrode locations")
            electrodes_df = dataset.get_electrode_locations(example_file)
            if electrodes_df is not None:
                eeg_logger.info(f"Found information for {len(electrodes_df)} channels")
                eeg_logger.info(f"Electrode sample:\n{electrodes_df.head()}")
        except Exception as e:
            logger.log_exception(eeg_logger, e, "Error reading electrode information")
        
        # Try to load the recording using MNE
        try:
            eeg_logger.info("Loading EEG data with MNE")
            raw = dataset.load_recording(example_file, preload=False)
            if raw is not None:
                eeg_logger.info(f"  Sampling rate: {raw.info['sfreq']} Hz")
                eeg_logger.info(f"  Channel count: {len(raw.ch_names)}")
                eeg_logger.info(f"  Recording duration: {raw.times[-1]:.1f} seconds")
                eeg_logger.info(f"  Channel types: {raw.get_channel_types()[:5]}...")
                
                # Sample channel names
                eeg_logger.info(f"  Channel names: {raw.ch_names[:5]}...")
        except Exception as e:
            logger.log_exception(eeg_logger, e, "Error loading recording with MNE")
            eeg_logger.info("  Note: This may be due to missing MNE-Python. Install with: pip install mne")

def main():
    # Define data directory - use an absolute path in the project root directory (not in examples)
    project_root = Path(__file__).parent.parent  # Go up one level to the project root
    
    # Set up logging first - using the centralized logging system
    logs_dir = project_root / "logs"
    log_file, error_log_file = logger.setup_logging(
        log_dir=logs_dir,
        log_file_prefix="eeg_example",
        force_flush=True
    )
    
    # Get main application logger
    app_logger = logger.get_logger('examples.main')
    
    # Print log file locations to console
    print(f"Log files:")
    print(f"  - Main log: {log_file}")
    print(f"  - Error log: {error_log_file}")
    
    # Start logging the app steps
    logger.log_step(app_logger, "Preprocessed EEG Dataset Example")
    
    # Set up data directory
    data_dir = project_root / "data"
    os.makedirs(data_dir, exist_ok=True)
    app_logger.info(f"Using data directory: {data_dir.absolute()}")
    
    # Initialize the dataset
    logger.log_step(app_logger, "Initializing dataset")
    dataset_id = "ds004408"  # EEG dataset with preprocessed data
    version = "1.0.0"        # Dataset version
    
    app_logger.info(f"Dataset ID: {dataset_id}, Version: {version}")
    logger.log_step(app_logger, "Downloading and initializing EEG dataset")
    
    # Use the factory function to get the EEG dataset handler
    # We can explicitly specify EEG type to avoid full download for detection
    dataset = create_dataset(
        dataset_id=dataset_id,
        data_dir=str(data_dir),
        version=version,
        force_type="eeg"  # Explicitly specify this is an EEG dataset
    )
    
    # Verify we got an EEG dataset handler
    app_logger.info(f"Dataset type: {type(dataset).__name__}")
    
    # Download the dataset before exploring
    logger.log_step(app_logger, "Downloading dataset")
    app_logger.info("This may take some time...")
    
    try:
        # No special DataLad configuration - we'll keep it simple
        print("\nStarting dataset download - this may take several minutes...")
        print("DataLad will show its own progress indicators if available.")
        
        # Perform the download
        download_success = dataset.download_dataset()
        
        if not download_success:
            app_logger.error("Failed to download dataset. Please check your internet connection and try again.")
            return
        
        app_logger.info("Dataset successfully downloaded.")
        
        # Verify dataset structure exists
        dataset_dir = Path(data_dir) / dataset_id
        if not dataset_dir.exists():
            app_logger.error(f"Expected dataset directory not found: {dataset_dir}")
            return
            
        # List some files to verify download
        app_logger.info("Verifying dataset structure:")
        try:
            for item in list(dataset_dir.iterdir())[:10]:  # List up to 10 items
                app_logger.info(f"  {item.relative_to(dataset_dir)}")
        except Exception as e:
            logger.log_exception(app_logger, e, "Error listing dataset contents")
    
    except Exception as e:
        logger.log_exception(app_logger, e, "Error during dataset download")
        return
    
    # Explore the dataset with EEG-specific methods
    explore_eeg_dataset(dataset)
    
    logger.log_step(app_logger, "Example Complete")
    app_logger.info(f"Dataset is now available in the {data_dir.absolute()} directory")
    app_logger.info("You can explore it further using EEGDataset methods")

if __name__ == "__main__":
    main() 