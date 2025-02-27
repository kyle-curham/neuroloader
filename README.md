# Neuroimaging Data Loader and Processor

A Python package for loading and processing neuroimaging data from various sources like OpenNeuro, with support for multiple modalities (MRI, fMRI, EEG).

## Features

- **Multi-Modal Support**: Works with MRI, fMRI, EEG data
- **Dataset Management**: Download and organize datasets from OpenNeuro and other repositories
- **Preprocessing Pipeline**: Modular preprocessing components with automatic derivative data detection
- **Events Handling**: Tools for working with experiment events and annotations
- **Data Integration**: Combine data from multiple sources and modalities
- **Automatic Pipeline Generation**: Create appropriate preprocessing pipelines based on data modalities and preprocessing status

## Installation

```bash
pip install -r requirements.txt
```

Make sure you have git-annex installed, which is required for DataLad to download datasets.

## Quick Start

```python
from neuroloader.factory import create_dataset
from neuroloader.processors.eeg_processor import EEGProcessor
from neuroloader.processors.pipeline import PreprocessingPipeline

# Create a dataset
dataset = create_dataset(
    "DS002778",  # OpenNeuro dataset ID
    data_dir="./data"
)

# Download the dataset
dataset.download_dataset()

# Check if this dataset is raw or derivative data
is_derivative = dataset.is_derivative()
print(f"Dataset is {'derivative' if is_derivative else 'raw'} data")

# Create a processor for EEG data
processor = EEGProcessor(dataset)

# Create a processing pipeline
pipeline = PreprocessingPipeline(name="my_pipeline", skip_if_derivative=True)

# Add processing steps
pipeline.add_processor(
    processor=processor,
    name="eeg_preprocess",
    params={
        "recording_file": dataset.get_recording_files('eeg')[0],
        "filter_params": {
            "l_freq": 1.0,
            "h_freq": 40.0,
            "notch_freqs": [50.0, 60.0]
        },
        "check_if_preprocessed": True  # Enable individual file-level checks
    }
)

# Execute the pipeline
results = pipeline.execute(datasets=[dataset])

# Check if preprocessing was skipped
if results.get("preprocessing_skipped", False):
    print("Preprocessing was skipped because dataset is derivative data")
else:
    print("Full preprocessing was applied to the raw dataset")
```

## Automatic Pipeline Generation

You can use the factory to automatically create both a dataset and an appropriate preprocessing pipeline:

```python
from neuroloader.factory import create_dataset

# Create a dataset and pipeline in one step
result = create_dataset(
    "ds000228",
    data_dir="./data",
    with_pipeline=True,
    pipeline_options={
        "skip_if_derivative": True,
        "mri_bias_correction": True,
        "fmri_motion_correction": True,
        "eeg_artifact_removal": True
    }
)

# Extract the dataset and pipeline
dataset = result["dataset"]
pipeline = result["pipeline"]

# Check the dataset modalities
modalities = dataset.get_modalities()
print(f"Detected modalities: {modalities}")

# Execute the pipeline
results = pipeline.execute(datasets=[dataset])
```

The factory will:
1. Detect available modalities in the dataset
2. Create appropriate processors for each modality
3. Configure preprocessing steps based on best practices
4. Set up the pipeline to respect derivative status
5. Allow customization through pipeline_options

## Working with Derivative Data

This package automatically detects whether a dataset contains raw or derivative (already preprocessed) data, and can adjust preprocessing steps accordingly:

1. **Dataset-Level Detection**: When initializing a dataset, it checks for indicators like the presence of a 'derivatives' directory or metadata in 'dataset_description.json'.

2. **File-Level Detection**: Each processor implements modality-specific checks to determine if individual files appear to be preprocessed.

3. **Pipeline Behavior**: The preprocessing pipeline can automatically skip preprocessing steps for derivative data while still executing analysis steps.

4. **Manual Override**: You can explicitly set derivative status if the automatic detection is incorrect.

```python
# Override derivative status
dataset.set_derivative_status(True)  # Mark as derivative/preprocessed data

# Force specific processing steps to run even on derivative data
pipeline.add_processor(
    processor=processor,
    name="compute_power_spectrum",
    params={...},
    force_execution=True  # This step runs regardless of derivative status
)
```

## Documentation

For detailed documentation, see the docstrings in the code and example scripts in the 'examples' directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 