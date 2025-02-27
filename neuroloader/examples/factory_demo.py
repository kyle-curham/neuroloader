"""
Factory Demo - Automatic Dataset Type Selection

This script demonstrates the automatic modality detection and appropriate handler
selection provided by the neuroloader.create_dataset factory function.

The script will:
1. Try creating datasets for different OpenNeuro dataset IDs
2. Show which type of handler was selected for each
3. Demonstrate forcing a specific handler type
4. Show how to download datasets with different handlers
"""

import sys
from pathlib import Path
import os

from neuroloader import create_dataset
from neuroloader import logger

# Configure logging using our centralized logger system
logger.configure_logger(
    level="INFO",
    handlers=[{"type": "stream", "stream": sys.stdout}]
)

# Get the package logger
demo_logger = logger.get_logger("examples.factory_demo")

def main():
    # Set up data directory
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    os.makedirs(data_dir, exist_ok=True)
    
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
        # Known multimodal dataset (EEG + fMRI)
        {"id": "ds002791", "version": "1.1.0", "expected_type": "Multimodal"},
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
            
            # Get a basic stat about the dataset
            info = dataset.describe()
            subject_count = info.get('subject_count', 'unknown')
            demo_logger.info(f"Subject count: {subject_count}")
            
            # Check if dataset is already downloaded
            if not dataset.is_downloaded():
                demo_logger.info(f"Dataset not yet downloaded. Use download_dataset() to download.")
            
        except ValueError as e:
            demo_logger.error(f"Modality detection error: {str(e)}")
        except Exception as e:
            demo_logger.error(f"Error: {str(e)}")
    
    # Demonstrate forcing a specific type
    demo_logger.info("\n--- Forcing Specific Handler Types ---")
    dataset_id = datasets[0]["id"]  # Use the first dataset
    
    for handler_type in ["multimodal", "eeg", "mri", "fmri"]:
        demo_logger.info(f"\nForcing {handler_type.upper()} handler for dataset {dataset_id}")
        try:
            dataset = create_dataset(
                dataset_id=dataset_id,
                data_dir=str(data_dir),
                version=datasets[0]["version"],
                force_type=handler_type
            )
            demo_logger.info(f"Selected handler: {type(dataset).__name__}")
            
            # Demonstrate download method (but don't actually download)
            demo_logger.info(f"To download: dataset.download_dataset(force=False)")
            
        except Exception as e:
            demo_logger.error(f"Error: {str(e)}")
    
    # Demonstrate downloading a small dataset
    try:
        demo_logger.info("\n--- Downloading a Small Dataset ---")
        # Choose a small dataset for the demo
        small_dataset_id = "ds003645"  # Small EEG dataset
        demo_logger.info(f"Attempting to download small dataset: {small_dataset_id}")
        
        small_dataset = create_dataset(
            dataset_id=small_dataset_id,
            data_dir=str(data_dir)
        )
        
        # Show download method (commenting out actual download to avoid 
        # downloading large files during demo unless explicitly requested)
        demo_logger.info("To download the dataset, you would call:")
        demo_logger.info("small_dataset.download_dataset(force=False)")
        demo_logger.info("# Downloading is commented out in this demo to avoid unexpected large downloads")
        # Uncomment to actually download:
        # if not small_dataset.is_downloaded():
        #     success = small_dataset.download_dataset(force=False)
        #     if success:
        #         demo_logger.info("Download successful!")
        #     else:
        #         demo_logger.info("Download failed.")
        
    except Exception as e:
        demo_logger.error(f"Download demo error: {str(e)}")
    
    demo_logger.info("\n===== Demo Complete =====")
    demo_logger.info("The factory function enables automatic selection of the appropriate")
    demo_logger.info("dataset handler based on the detected modalities, or you can")
    demo_logger.info("force a specific handler when needed.")
    demo_logger.info("\nAll handlers now consistently support download_dataset() method")
    demo_logger.info("and is_downloaded() check due to inheritance from BaseDataset.")

if __name__ == "__main__":
    main() 