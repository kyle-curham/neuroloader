"""
Example script for downloading and exploring a neuroimaging dataset using DataLad API.

This example demonstrates:
1. Initializing a dataset with automatic modality detection
2. Downloading it using DataLad's Python API
3. Exploring the dataset contents based on its type
"""

from neuroloader import create_dataset
from neuroloader.multimodal import MultimodalDataset
from neuroloader.mri import MRIDataset, FMRIDataset
from neuroloader.eeg import EEGDataset
import logging
import sys
import os
from pathlib import Path
from typing import Any

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def explore_dataset(dataset: Any) -> None:
    """
    Explore the contents of a dataset, adapting to its type.
    
    Args:
        dataset: The dataset object to explore
    """
    # Get dataset information
    print("\n3. Dataset Information")
    dataset_info = dataset.describe()
    
    if hasattr(dataset_info, 'get'):
        subject_count = dataset_info.get('subject_count', 'unknown')
        subjects = dataset_info.get('subjects', [])
        print(f"Dataset contains {subject_count} subjects")
        if subjects:
            print(f"Subject IDs: {subjects[:5]}...")
    
    # For multimodal datasets, show modality information
    if isinstance(dataset, MultimodalDataset):
        # Check available modalities
        print("\n4. Available Modalities")
        available = [m for m, avail in dataset.available_modalities.items() if avail]
        print(f"Available modalities: {available}")
        
        # Get files by modality
        print("\n5. Files by Modality (first 3 of each)")
        for modality in available:
            if modality == "eeg":
                files = dataset.get_eeg_files()
                print(f"EEG files: {len(files)} found")
                for f in files[:3]:
                    print(f"  - {f}")
            elif modality == "mri":
                files = dataset.get_mri_files()
                print(f"MRI files: {len(files)} found")
                for f in files[:3]:
                    print(f"  - {f}")
            elif modality == "fmri":
                files = dataset.get_fmri_files()
                print(f"fMRI files: {len(files)} found")
                for f in files[:3]:
                    print(f"  - {f}")
        
        # Find multimodal runs (sessions with multiple imaging types)
        print("\n6. Multimodal Runs (first 2)")
        multimodal_runs = dataset.get_multimodal_runs()
        print(f"Found {len(multimodal_runs)} multimodal runs")
        
        for i, (run_key, modality_files) in enumerate(list(multimodal_runs.items())[:2]):
            print(f"\nRun {i+1}: {run_key}")
            for modality, files in modality_files.items():
                if files:
                    print(f"  {modality}: {len(files)} files")
    
    # For EEG datasets, show specific EEG information
    elif isinstance(dataset, EEGDataset):
        files = dataset.get_recording_files()
        print("\n4. EEG Files")
        print(f"Found {len(files)} EEG recording files")
        for f in files[:3]:
            print(f"  - {f}")
            
        # Show event information for the first file if available
        if files:
            print("\n5. Example Events Information")
            try:
                events = dataset.get_events_dataframe(files[0])
                print(f"Events in {files[0].name}:")
                print(events.head())
            except Exception as e:
                print(f"Could not read events: {e}")
    
    # For MRI datasets, show specific MRI information
    elif isinstance(dataset, MRIDataset):
        # Show structural scans
        t1_scans = dataset.get_structural_scans("T1w")
        t2_scans = dataset.get_structural_scans("T2w")
        
        print("\n4. Structural MRI Scans")
        print(f"T1-weighted scans: {len(t1_scans)} found")
        for f in t1_scans[:3]:
            print(f"  - {f}")
            
        print(f"\nT2-weighted scans: {len(t2_scans)} found")
        for f in t2_scans[:3]:
            print(f"  - {f}")
    
    # For fMRI datasets, show specific fMRI information
    elif isinstance(dataset, FMRIDataset):
        # Show functional scans
        func_scans = dataset.get_functional_scans()
        
        print("\n4. Functional MRI Scans")
        print(f"Found {len(func_scans)} functional scans")
        for f in func_scans[:3]:
            print(f"  - {f}")
            
        # Show task information if available
        tasks = dataset.get_available_tasks()
        if tasks:
            print("\n5. Available Tasks")
            for task in tasks:
                print(f"  - {task}")

def main():
    print("====== Neuroimaging Dataset Example ======")
    
    # Define data directory - use an absolute path within the neuroloader project
    # Get the current script directory and go up one level to project root
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    
    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Using data directory: {data_dir.absolute()}")
    
    # Initialize the dataset
    print("\n1. Initializing dataset")
    dataset_id = "ds005907"  # Dataset ID from OpenNeuro
    version = "1.0.0"        # Dataset version
    
    print(f"Dataset ID: {dataset_id}, Version: {version}")
    print("\n2. Downloading and determining dataset type")
    
    # Use the factory function to get the appropriate dataset handler
    # based on automatic modality detection
    dataset = create_dataset(
        dataset_id=dataset_id,
        data_dir=str(data_dir),
        version=version
    )
    
    # Print the type of dataset we're working with
    print(f"Dataset type: {type(dataset).__name__}")
    
    # Explore the dataset with the appropriate handler
    explore_dataset(dataset)
    
    print("\n====== Example Complete ======")
    print(f"Dataset is now available in the {data_dir.absolute()} directory")
    print("You can explore it further using the appropriate dataset methods")

if __name__ == "__main__":
    main() 