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
    
    # Extract subject information
    logger.log_step(eeg_logger, "Exploring subject information")
    subject_ids = dataset_info.get('subjects', [])
    eeg_logger.info(f"Found {len(subject_ids)} subjects: {subject_ids}")
    
    # For each subject, get their files
    for subject_id in subject_ids[:3]:  # Limit to first 3 subjects to avoid too much output
        subject_files = dataset.get_subject_files(subject_id)
        eeg_logger.info(f"Subject {subject_id} has {len(subject_files)} files")
        
        # Filter for only EEG files
        eeg_files = [f for f in subject_files if f.suffix in ['.set', '.edf', '.bdf', '.vhdr', '.cnt', '.eeg']]
        eeg_logger.info(f"  Of which {len(eeg_files)} are EEG recordings")
        
        # Show the first recording for this subject
        if eeg_files:
            eeg_logger.info(f"  First recording: {eeg_files[0].name}")
    
    # Check if there are any recordings
    if recording_files:
        # Get the first recording file for exploration
        example_file = recording_files[0]
        eeg_logger.info(f"Exploring example EEG file: {example_file.name}")
        
        # Get events for the example recording
        try:
            logger.log_step(eeg_logger, "Extracting events information")
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
                elif 'value' in events_df.columns:
                    event_counts = events_df['value'].value_counts()
                    eeg_logger.info("Event Value Counts:")
                    for event_type, count in event_counts.items():
                        eeg_logger.info(f"  {event_type}: {count}")
            else:
                eeg_logger.warning("No events found in the recording. Trying alternative methods...")
                
                # Try to find events in other recordings
                for alt_file in recording_files[1:3]:  # Try a few more files
                    eeg_logger.info(f"Trying to extract events from alternative file: {alt_file.name}")
                    alt_events_df = dataset.get_events_dataframe(alt_file)
                    if not alt_events_df.empty:
                        eeg_logger.info(f"Found {len(alt_events_df)} events in alternative file")
                        eeg_logger.info(f"Events sample:\n{alt_events_df.head()}")
                        break
        except Exception as e:
            logger.log_exception(eeg_logger, e, "Error reading events")
        
        # Try to get electrode locations
        try:
            logger.log_step(eeg_logger, "Extracting electrode information")
            electrodes_df = dataset.get_electrode_locations(example_file)
            if electrodes_df is not None:
                eeg_logger.info(f"Found information for {len(electrodes_df)} channels")
                eeg_logger.info(f"Electrode sample:\n{electrodes_df.head()}")
                
                # Check if coordinates are available
                if all(col in electrodes_df.columns for col in ['x', 'y', 'z']):
                    eeg_logger.info("3D electrode coordinates are available")
                elif all(col in electrodes_df.columns for col in ['x', 'y']):
                    eeg_logger.info("2D electrode coordinates are available")
                
                # Check for impedance information
                if 'impedance' in electrodes_df.columns:
                    impedances = electrodes_df['impedance'].dropna()
                    if not impedances.empty:
                        eeg_logger.info(f"Impedance information available for {len(impedances)} channels")
                        eeg_logger.info(f"Average impedance: {impedances.mean():.2f} kOhm")
            else:
                eeg_logger.warning("No electrode location information found")
                
                # Try with another file
                for alt_file in recording_files[1:3]:
                    eeg_logger.info(f"Trying to extract electrode information from alternative file: {alt_file.name}")
                    alt_electrodes_df = dataset.get_electrode_locations(alt_file)
                    if alt_electrodes_df is not None:
                        eeg_logger.info(f"Found information for {len(alt_electrodes_df)} channels in alternative file")
                        eeg_logger.info(f"Electrode sample:\n{alt_electrodes_df.head()}")
                        break
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
    logger.log_step(app_logger, "EEG Dataset Example")
    
    # Set up data directory
    data_dir = project_root / "data"
    os.makedirs(data_dir, exist_ok=True)
    app_logger.info(f"Using data directory: {data_dir.absolute()}")
    
    # Initialize the dataset
    logger.log_step(app_logger, "Initializing dataset")
    # Use the new dataset with better BIDS compliance
    dataset_id = "ds004448"  # SMR-BCI dataset with standard BIDS format
    version = "1.0.1"        # Dataset version
    
    app_logger.info(f"Dataset ID: {dataset_id}, Version: {version}")
    
    # Use the factory function to get the appropriate dataset handler
    # The create_dataset function will now detect modalities before downloading
    logger.log_step(app_logger, "Detecting dataset modality")
    try:
        dataset = create_dataset(
            dataset_id=dataset_id,
            data_dir=str(data_dir),
            version=version
        )
        
        # Verify we got the correct dataset handler
        app_logger.info(f"Dataset type: {type(dataset).__name__}")
    except ValueError as e:
        app_logger.error(f"Failed to detect modalities: {str(e)}")
        print(f"\nERROR: {str(e)}")
        print("You can force a specific dataset type by using force_type parameter:")
        print("  dataset = create_dataset(dataset_id, data_dir, force_type='eeg')")
        return
    
    # Download the dataset if needed
    logger.log_step(app_logger, "Downloading dataset if needed")
    app_logger.info("This may take some time...")
    
    # Verify dataset structure exists
    dataset_dir = Path(data_dir) / dataset_id
    if not dataset_dir.exists() or len(list(dataset_dir.glob("*"))) == 0:
        try:
            print("\nStarting dataset download - this may take several minutes...")
            print("DataLad will show its own progress indicators if available.")
            
            # Perform the download
            download_success = dataset.download_dataset()
            
            if not download_success:
                app_logger.error("Failed to download dataset. Please check your internet connection and try again.")
                return
            
            app_logger.info("Dataset successfully downloaded.")
            
        except Exception as e:
            logger.log_exception(app_logger, e, "Error during dataset download")
            return
    else:
        app_logger.info("Dataset already downloaded. Skipping download step.")
    
    # Verify dataset structure
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
    
    # Ensure we're working with the correct dataset type for EEG-specific exploration
    if isinstance(dataset, EEGDataset):
        # Explore the dataset with EEG-specific methods
        explore_eeg_dataset(dataset)
    else:
        app_logger.warning(f"Expected EEG dataset but got {type(dataset).__name__}. Cannot run EEG-specific exploration.")
    
    logger.log_step(app_logger, "Example Complete")
    app_logger.info(f"Dataset is now available in the {data_dir.absolute()} directory")
    app_logger.info("You can explore it further using the dataset methods")

if __name__ == "__main__":
    main() 