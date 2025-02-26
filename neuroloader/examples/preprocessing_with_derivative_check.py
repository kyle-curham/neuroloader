"""Example of using the preprocessing pipeline with derivative data checks.

This script demonstrates how to:
1. Load a dataset from OpenNeuro
2. Check if it's raw or derivative data
3. Set up a preprocessing pipeline that respects derivative status
4. Run the pipeline on the dataset
"""

import os
import logging
from pathlib import Path
import sys

# Add the project root to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from neuroloader.factory import DatasetFactory
from neuroloader.processors.eeg_processor import EEGProcessor
from neuroloader.processors.mri_processor import MRIProcessor
from neuroloader.processors.fmri_processor import FMRIProcessor
from neuroloader.processors.pipeline import PreprocessingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the example."""
    # Dataset ID from OpenNeuro (example: DS002778 is an EEG dataset)
    # You can replace with any valid OpenNeuro dataset ID
    dataset_id = "DS002778"  # EEG dataset
    
    # Create data directory if it doesn't exist
    data_dir = Path("./data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create a dataset using the factory
    logger.info(f"Loading dataset {dataset_id}")
    dataset = DatasetFactory.create_dataset(dataset_id, data_dir=data_dir)
    
    # Ensure the dataset is downloaded
    if not dataset.is_downloaded():
        logger.info(f"Downloading dataset {dataset_id}")
        dataset.download_dataset()
    else:
        logger.info(f"Dataset {dataset_id} already downloaded")
    
    # Check if the dataset is derivative data
    is_derivative = dataset.is_derivative()
    logger.info(f"Dataset {dataset_id} is {'derivative' if is_derivative else 'raw'} data")
    
    # Manual override example (uncomment to test)
    # dataset.set_derivative_status(not is_derivative)  # Toggle for testing
    
    # Create processors based on the dataset type
    modalities = dataset.get_modalities()
    logger.info(f"Dataset modalities: {modalities}")
    
    # Create appropriate processors based on dataset modalities
    processors = []
    if 'eeg' in modalities:
        processors.append(('eeg', EEGProcessor(dataset)))
    if 'T1w' in modalities:
        processors.append(('mri', MRIProcessor(dataset)))
    if 'bold' in modalities:
        processors.append(('fmri', FMRIProcessor(dataset)))
    
    if not processors:
        logger.warning("No supported modalities found in dataset")
        return
    
    # Create preprocessing pipeline
    pipeline = PreprocessingPipeline(name="example_pipeline", skip_if_derivative=True)
    
    # Add processing steps for each modality
    for modality_name, processor in processors:
        if modality_name == 'eeg':
            # Get the first EEG recording file
            eeg_files = dataset.get_recording_files('eeg')
            if eeg_files:
                # Add preprocessing step for the EEG file
                pipeline.add_processor(
                    processor=processor,
                    name="eeg_preprocess",
                    params={
                        "recording_file": eeg_files[0],
                        "filter_params": {
                            "l_freq": 1.0,
                            "h_freq": 40.0,
                            "notch_freqs": [50.0, 60.0]
                        },
                        "check_if_preprocessed": True  # Enable file-level derivative check
                    }
                )
                
                # Add a power spectrum computation step that runs even for derivative data
                pipeline.add_processor(
                    processor=processor,
                    name="compute_psd",
                    params={
                        "method": "welch",
                        "fmin": 1,
                        "fmax": 40
                    },
                    input_keys={"epochs": "epochs"},  # Use the epochs from previous step
                    force_execution=True  # This step will run even for derivative data
                )
        
        elif modality_name == 'mri':
            # Get the first T1w file
            mri_files = dataset.get_recording_files('T1w')
            if mri_files:
                pipeline.add_processor(
                    processor=processor,
                    name="mri_preprocess",
                    params={
                        "scan_file": mri_files[0],
                        "bias_correction": True,
                        "skull_stripping": True
                    }
                )
        
        elif modality_name == 'fmri':
            # Get the first BOLD file
            fmri_files = dataset.get_recording_files('bold')
            if fmri_files:
                pipeline.add_processor(
                    processor=processor,
                    name="fmri_preprocess",
                    params={
                        "scan_file": fmri_files[0],
                        "motion_correction": True,
                        "slice_timing": True
                    }
                )
    
    # Execute the pipeline
    logger.info("Executing preprocessing pipeline")
    results = pipeline.execute(datasets=[dataset])
    
    # Check if preprocessing was skipped due to derivative data
    if results.get("preprocessing_skipped", False):
        logger.info("Preprocessing was skipped because dataset is derivative data")
        logger.info("Only steps with force_execution=True were executed")
    else:
        logger.info("Full preprocessing was applied to the raw dataset")
    
    # Print execution history
    logger.info("Pipeline execution history:")
    for step in pipeline.execution_history:
        status = step.get("status", "unknown")
        name = step.get("step_name", "unknown")
        duration = step.get("duration_seconds", 0)
        logger.info(f"  - {name}: {status} ({duration:.2f}s)")
    
    logger.info("Example completed successfully")

if __name__ == "__main__":
    main() 