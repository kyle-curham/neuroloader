#!/usr/bin/env python3
"""Example of using the factory to create a dataset and preprocessing pipeline.

This example shows how to use the factory module to create the appropriate dataset
handler based on the detected modalities and a preprocessing pipeline tailored to
those modalities, taking into account whether the data is raw or derivative.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path for the example to work from any location
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuroloader.factory import create_dataset
from neuroloader.logger import get_logger

# Setup logger for this application
app_logger = get_logger("preprocessing_example")

def main():
    """Main function to run the example."""
    # Setup argument parser for the example
    parser = argparse.ArgumentParser(description="Preprocessing pipeline example")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="ds000228", 
        help="Dataset ID from OpenNeuro (default: ds000228)"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="./data", 
        help="Directory to store the data (default: ./data)"
    )
    parser.add_argument(
        "--force-type", 
        type=str, 
        choices=["multimodal", "eeg", "mri", "fmri"], 
        help="Force a specific dataset type"
    )
    parser.add_argument(
        "--execute-pipeline", 
        action="store_true", 
        help="Execute the preprocessing pipeline"
    )
    parser.add_argument(
        "--skip-derivative", 
        action="store_true", 
        help="Skip preprocessing for derivative datasets"
    )
    
    args = parser.parse_args()
    
    # Get the dataset ID and data directory from arguments
    dataset_id = args.dataset
    data_dir = Path(args.data_dir)
    force_type = args.force_type
    execute_pipeline = args.execute_pipeline
    skip_derivative = args.skip_derivative
    
    app_logger.info(f"Using data directory: {data_dir.absolute()}")
    
    # Define pipeline options
    pipeline_options = {
        "skip_if_derivative": skip_derivative,
        # EEG options
        "eeg_filtering": True,
        "eeg_resampling": True,
        "eeg_artifact_removal": True,
        # MRI options
        "mri_bias_correction": True,
        "mri_skull_stripping": True, 
        "mri_segmentation": True,
        # fMRI options
        "fmri_motion_correction": True,
        "fmri_slice_timing": True,
        "fmri_spatial_smoothing": 6.0
    }
    
    # Create a dataset and pipeline using the factory function
    app_logger.info(f"Creating dataset and pipeline for {dataset_id}")
    
    try:
        # Use the create_dataset function with pipeline generation
        result = create_dataset(
            dataset_id, 
            data_dir=data_dir, 
            force_type=force_type,
            with_pipeline=True,
            pipeline_options=pipeline_options
        )
        
        # Extract dataset and pipeline
        dataset = result["dataset"]
        pipeline = result["pipeline"]
        
        # Display dataset information
        app_logger.info(f"Created dataset: {type(dataset).__name__}")
        
        # Check if pipeline was created
        if pipeline is None:
            app_logger.warning("No preprocessing pipeline could be created for this dataset")
            return
        
        app_logger.info(f"Created pipeline with {len(pipeline.steps)} preprocessing steps:")
        for i, step in enumerate(pipeline.steps):
            app_logger.info(f"  Step {i+1}: {step['name']} ({step['type']})")
        
        # Check if the dataset is derivative data
        is_derivative = dataset.is_derivative()
        app_logger.info(f"Dataset {dataset_id} is {'derivative' if is_derivative else 'raw'} data")
        
        # Ensure the dataset is downloaded (required before executing pipeline)
        if execute_pipeline and not dataset.is_downloaded():
            app_logger.info(f"Downloading dataset {dataset_id}")
            download_success = dataset.download_dataset()
            if not download_success:
                app_logger.error("Failed to download dataset. Please check your internet connection and try again.")
                return
            app_logger.info("Dataset successfully downloaded.")
        
        # Execute the pipeline if requested
        if execute_pipeline:
            app_logger.info("Executing preprocessing pipeline...")
            pipeline_results = pipeline.execute(datasets=[dataset])
            
            # Check if preprocessing was skipped due to derivative data
            if pipeline_results.get("preprocessing_skipped", False):
                app_logger.info("Preprocessing was skipped because dataset is derivative data")
            else:
                app_logger.info("Preprocessing completed successfully")
                
            # Save the execution history and results
            results_dir = data_dir / "results"
            os.makedirs(results_dir, exist_ok=True)
            
            # Save pipeline results
            results_file = results_dir / f"{dataset_id}_pipeline_results.json"
            pipeline.save_results(results_file)
            app_logger.info(f"Saved pipeline results to {results_file}")
            
            # Save preprocessing history
            history = pipeline.get_preprocessing_history()
            app_logger.info(f"Pipeline executed {len(history)} preprocessing actions")
        
    except Exception as e:
        app_logger.error(f"Error: {str(e)}", exc_info=True)
        return

if __name__ == "__main__":
    main() 