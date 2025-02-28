"""
Factory Demo - Automatic Dataset Type Selection

This script demonstrates the automatic modality detection and appropriate handler
selection provided by the neuroloader.create_dataset factory function.

The script will:
1. Try creating datasets for different OpenNeuro dataset IDs
2. Show which type of handler was selected for each
3. Demonstrate forcing a specific handler type
4. Download a small dataset and show detailed information
5. Demonstrate creating datasets with preprocessing pipelines
"""

import sys
from pathlib import Path
import os
import logging
import json

from neuroloader import create_dataset
from neuroloader import logger

# Set up the project root and directories
project_root = Path(__file__).parent.parent  # Go up one level to neuroloader/ root
logs_dir = project_root / "logs"
data_dir = project_root / "data"  # This puts data in neuroloader/data instead of neuroloader/examples/data
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Configure logging using our centralized logger system
log_file, error_log_file = logger.setup_logging(
    console_level=logging.INFO,
    file_level=logging.DEBUG,  # Log more details to file
    log_dir=logs_dir,  # Enable file logging
    log_file_prefix="factory_demo",
    propagate_to_root=False
)

# Log the file locations for reference
print(f"Logging to file: {log_file}")
print(f"Error logging to: {error_log_file}")

# Get the package logger
demo_logger = logger.get_logger("examples.factory_demo")

def main():
    # Log start of demo
    demo_logger.info("===== Factory Demo - Automatic Dataset Type Selection =====")
    demo_logger.info(f"Using data directory: {data_dir.absolute()}\n")
    
    # Define some datasets to test with different modalities
    datasets = [
        # Known EEG dataset
        {"id": "ds002778", "version": "1.0.3", "expected_type": "EEG"},
        # Known fMRI dataset
        {"id": "ds000117", "version": "1.0.4", "expected_type": "fMRI"},
        # Known structural MRI dataset
        {"id": "ds000113", "version": "1.0.0", "expected_type": "MRI"},
        # Known multimodal dataset (EEG + MRI) - replacing unavailable ds002791
        {"id": "ds003645", "version": "1.0.1", "expected_type": "Multimodal"},
    ]
    
    # Test automatic detection for each dataset
    demo_logger.info("\n--- Automatic Detection ---")
    for i, dataset_info in enumerate(datasets):
        dataset_id = dataset_info['id']
        demo_logger.info(f"\nDataset {i+1}: {dataset_id} (expected: {dataset_info['expected_type']})")
        try:
            dataset = create_dataset(
                dataset_id=dataset_id,
                data_dir=str(data_dir),
                version=dataset_info["version"]
            )
            demo_logger.info(f"Selected handler: {type(dataset).__name__}")
            
            # Check if dataset is already downloaded
            if dataset.is_downloaded():
                demo_logger.info(f"Dataset is already downloaded. Getting full information...")
                info = dataset.describe()
                demo_logger.info(f"Subject count: {info.get('subject_count', 'unknown')}")
            else:
                # Following the pattern from EEG loader, just display basic info
                # without attempting to access files that don't exist yet
                demo_logger.info(f"Dataset not downloaded. Only basic information available.")
                info = dataset.describe()
                demo_logger.info(f"Dataset ID: {info.get('dataset_id')}")
                demo_logger.info(f"Dataset type: {info.get('dataset_type')}")
                demo_logger.info(f"Download status: {info.get('download_status')}")
            
        except ValueError as e:
            demo_logger.error(f"Modality detection error: {str(e)}")
        except Exception as e:
            demo_logger.error(f"Error: {str(e)}")
    
    # Demonstrate forcing a specific type
    demo_logger.info("\n--- Forcing Specific Handler Types ---")
    dataset_id = datasets[0]["id"]  # Use the first dataset
    
    for handler_type in ["eeg", "mri", "fmri"]:
        demo_logger.info(f"\nForcing {handler_type.upper()} handler for dataset {dataset_id}")
        try:
            dataset = create_dataset(
                dataset_id=dataset_id,
                data_dir=str(data_dir),
                version=datasets[0]["version"],
                force_type=handler_type
            )
            demo_logger.info(f"Selected handler: {type(dataset).__name__}")
            
        except Exception as e:
            demo_logger.error(f"Error: {str(e)}")
    
    # Download and demonstrate a small dataset
    try:
        demo_logger.info("\n--- Downloading a Small Dataset ---")
        # Choose a very small dataset for the demo that won't take too long to download
        small_dataset_id = "ds002720"  # Small EEG dataset (~50MB instead of 114GB)
        demo_logger.info(f"Downloading small dataset: {small_dataset_id}")
        
        small_dataset = create_dataset(
            dataset_id=small_dataset_id,
            data_dir=str(data_dir)
        )
        
        # Show basic info before downloading
        demo_logger.info("\nBefore download, describe() returns limited information:")
        pre_download_info = small_dataset.describe()
        demo_logger.info(f"Dataset ID: {pre_download_info.get('dataset_id')}")
        demo_logger.info(f"Dataset type: {pre_download_info.get('dataset_type')}")
        demo_logger.info(f"Download status: {pre_download_info.get('download_status')}")
        
        # Actually download the dataset
        if not small_dataset.is_downloaded():
            demo_logger.info("\nDownloading dataset (this may take a few minutes)...")
            success = small_dataset.download_dataset(force=False)
            if success:
                demo_logger.info("Download successful!")
                
                # Now get and show the full dataset information
                demo_logger.info("\nAfter download, describe() returns full dataset information:")
                full_info = small_dataset.describe()
                
                # Pretty print the full info
                demo_logger.info(f"Dataset ID: {full_info.get('dataset_id')}")
                demo_logger.info(f"Dataset type: {full_info.get('dataset_type')}")
                demo_logger.info(f"Download status: {full_info.get('download_status')}")
                demo_logger.info(f"Subject count: {full_info.get('subject_count', 'unknown')}")
                
                # Print subjects if available
                if 'subjects' in full_info:
                    demo_logger.info(f"Subjects: {', '.join(full_info.get('subjects', []))}")
                
                # Print modality-specific information
                if 'eeg_info' in full_info:
                    eeg_info = full_info['eeg_info']
                    demo_logger.info("\nEEG-specific information:")
                    demo_logger.info(f"Recording count: {eeg_info.get('recording_count', 'unknown')}")
                    if 'recording_formats' in eeg_info:
                        formats = [f"{ext} ({count})" for ext, count in eeg_info['recording_formats'].items()]
                        demo_logger.info(f"Recording formats: {', '.join(formats)}")
                    
                    # Show a condensed version of sample recording info
                    if 'sample_recording_info' in eeg_info:
                        sample_info = eeg_info['sample_recording_info']
                        if sample_info:
                            demo_logger.info("\nSample recording information:")
                            if 'channel_count' in sample_info:
                                demo_logger.info(f"Channel count: {sample_info.get('channel_count')}")
                            if 'sfreq' in sample_info:
                                demo_logger.info(f"Sampling frequency: {sample_info.get('sfreq')} Hz")
                
                # If it's MRI/fMRI, show those details
                if 'mri_info' in full_info:
                    mri_info = full_info['mri_info']
                    demo_logger.info("\nMRI-specific information:")
                    demo_logger.info(f"Scan count: {mri_info.get('scan_count', 'unknown')}")
                    if 'scan_types' in mri_info:
                        types = [f"{type_name} ({count})" for type_name, count in mri_info['scan_types'].items()]
                        demo_logger.info(f"Scan types: {', '.join(types)}")
            else:
                demo_logger.error("Download failed. Unable to show full dataset information.")
        else:
            demo_logger.info("Dataset already downloaded.")
            full_info = small_dataset.describe()
            demo_logger.info(f"Subject count: {full_info.get('subject_count', 'unknown')}")
        
    except Exception as e:
        demo_logger.error(f"Download demo error: {str(e)}")
        
    # Demonstrate the new with_pipeline parameter 
    try:
        demo_logger.info("\n--- Creating Dataset with Preprocessing Pipeline ---")
        pipeline_dataset_id = "ds002778"  # Use an EEG dataset
        demo_logger.info(f"Creating dataset with pipeline for: {pipeline_dataset_id}")
        
        # Define some pipeline options
        pipeline_options = {
            "eeg": {
                "bandpass_filter": {"l_freq": 1.0, "h_freq": 40.0},
                "notch_filter": {"freqs": [50, 100]}
            }
        }
        
        # Create dataset with pipeline - returns a dictionary with 'dataset' and 'pipeline' keys
        result = create_dataset(
            dataset_id=pipeline_dataset_id,
            data_dir=str(data_dir),
            with_pipeline=True,
            pipeline_options=pipeline_options
        )
        
        # Extract components from the result dictionary
        dataset = result.get("dataset")
        pipeline = result.get("pipeline")
        
        if dataset and pipeline:
            demo_logger.info(f"Created dataset of type: {type(dataset).__name__}")
            demo_logger.info(f"Created pipeline of type: {type(pipeline).__name__}")
            
            # Show pipeline capabilities
            demo_logger.info("\nThe pipeline can be used to process the dataset after downloading:")
            demo_logger.info("1. download_dataset()")
            demo_logger.info("2. processed_data = pipeline.run(dataset)")
        else:
            demo_logger.warning("Pipeline or dataset is missing from the result")
        
    except Exception as e:
        demo_logger.error(f"Pipeline demo error: {str(e)}")
    
    demo_logger.info("\n===== Demo Complete =====")
    demo_logger.info("The factory function enables automatic selection of the appropriate")
    demo_logger.info("dataset handler based on the detected modalities, or you can")
    demo_logger.info("force a specific handler when needed.")
    demo_logger.info("\nNOTE: While basic information can be retrieved without downloading,")
    demo_logger.info("detailed information and file access requires downloading the dataset first.")
    demo_logger.info("\nThe factory can now also create preprocessing pipelines appropriate")
    demo_logger.info("for the detected modalities using the with_pipeline parameter.")
    demo_logger.info("\nAll handlers consistently support download_dataset() method")
    demo_logger.info("and is_downloaded() check due to inheritance from BaseDataset.")

if __name__ == "__main__":
    main() 