"""
Factory Demo - Automatic Dataset Type Selection

This script demonstrates the automatic modality detection and appropriate handler
selection provided by the neuroloader.create_dataset factory function.

The script will:
1. Try creating datasets for different OpenNeuro dataset IDs
2. Show which type of handler was selected for each
3. Demonstrate forcing a specific handler type
"""

import logging
import sys
from pathlib import Path
import os

from neuroloader import create_dataset

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    # Set up data directory
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("===== Factory Demo - Automatic Dataset Type Selection =====")
    print(f"Using data directory: {data_dir.absolute()}\n")
    
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
    print("\n--- Automatic Detection ---")
    for i, dataset_info in enumerate(datasets):
        print(f"\nDataset {i+1}: {dataset_info['id']} (expected: {dataset_info['expected_type']})")
        try:
            dataset = create_dataset(
                dataset_id=dataset_info["id"],
                data_dir=str(data_dir),
                version=dataset_info["version"]
            )
            print(f"Selected handler: {type(dataset).__name__}")
            
            # Get a basic stat about the dataset
            if hasattr(dataset, 'describe'):
                info = dataset.describe()
                if info and hasattr(info, 'get'):
                    subject_count = info.get('subject_count', 'unknown')
                    print(f"Subject count: {subject_count}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Demonstrate forcing a specific type
    print("\n--- Forcing Specific Handler Types ---")
    dataset_id = datasets[0]["id"]  # Use the first dataset
    
    for handler_type in ["multimodal", "eeg", "mri", "fmri"]:
        print(f"\nForcing {handler_type.upper()} handler for dataset {dataset_id}")
        try:
            dataset = create_dataset(
                dataset_id=dataset_id,
                data_dir=str(data_dir),
                version=datasets[0]["version"],
                force_type=handler_type
            )
            print(f"Selected handler: {type(dataset).__name__}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n===== Demo Complete =====")
    print("The factory function enables automatic selection of the appropriate")
    print("dataset handler based on the detected modalities, or you can")
    print("force a specific handler when needed.")

if __name__ == "__main__":
    main() 