# Neuroimaging Data Manager

A Python package for downloading, processing, and analyzing neuroimaging data from OpenNeuro and other sources.

## Features

- Download neuroimaging datasets from OpenNeuro using DataLad
- Support for various data types:
  - EEG (EEGLab, BrainVision, EDF, BDF)
  - MRI (T1, T2 structural scans)
  - fMRI (BOLD)
- Extract recording filenames and metadata
- Create events DataFrames for experiments
- Process electrode locations for EEG
- **Handle multimodal datasets** with coordinated access to different imaging types

## Installation

```bash
# Clone the repository
git clone https://github.com/kyle-curham/neuroloader.git
cd neuroloader

# Install dependencies
pip install -r requirements.txt

# Install DataLad (required for downloading datasets)
pip install datalad

# Install the package in development mode
pip install -e .
```

## Usage Examples

### Download an EEG Dataset

```python
from neuroloader.eeg import EEGDataset

# Initialize dataset with OpenNeuro dataset ID
eeg_dataset = EEGDataset(
    dataset_id="ds003645",  # Example dataset ID
    data_dir="./data"       # Local directory to store data
)

# Download the dataset (uses DataLad)
eeg_dataset.download_dataset()

# Get all EEG recording files
recording_files = eeg_dataset.get_recording_files()
print(f"Found {len(recording_files)} EEG recordings")

# Get events for the first recording
if recording_files:
    events_df = eeg_dataset.get_events_dataframe(recording_files[0])
    print(events_df.head())

    # Get electrode locations for the recording
    electrode_df = eeg_dataset.get_electrode_locations(recording_files[0])
    if electrode_df is not None:
        print(electrode_df.head())
```

### Work with MRI/fMRI Data

```python
from neuroloader.mri import MRIDataset, FMRIDataset

# Initialize an MRI dataset
mri_dataset = MRIDataset(
    dataset_id="ds001393",  # Example dataset ID
    data_dir="./data"       # Local directory to store data
)

# Download the dataset (uses DataLad)
mri_dataset.download_dataset()

# Get T1-weighted structural scans
t1_scans = mri_dataset.get_structural_scans(scan_type="T1w")
print(f"Found {len(t1_scans)} T1-weighted scans")

# Load the first scan
if t1_scans:
    img, metadata = mri_dataset.load_scan(t1_scans[0])
    if img is not None:
        print(f"Loaded scan with shape {img.shape}")
        if metadata:
            print(f"Scan metadata: {metadata}")

# Initialize an fMRI dataset
fmri_dataset = FMRIDataset(
    dataset_id="ds002422",  # Example dataset ID
    data_dir="./data"       # Local directory to store data
)

# Download the dataset (uses DataLad)
fmri_dataset.download_dataset()

# Get all functional scans
func_scans = fmri_dataset.get_functional_scans()
print(f"Found {len(func_scans)} functional scans")

# Get events for the first functional scan
if func_scans:
    events_df = fmri_dataset.get_events_dataframe(func_scans[0])
    print(events_df.head())
```

### Work with Multimodal Data

```python
from neuroloader.multimodal import MultimodalDataset

# Initialize a multimodal dataset
multimodal_dataset = MultimodalDataset(
    dataset_id="ds003505",  # Example dataset with multiple modalities
    data_dir="./data"       # Local directory to store data
)

# Download the dataset (uses DataLad)
multimodal_dataset.download_dataset()

# Check available modalities
print(f"Available modalities: {multimodal_dataset.available_modalities}")

# Get subject IDs
subject_ids = multimodal_dataset.get_subject_ids()
print(f"Dataset contains {len(subject_ids)} subjects: {subject_ids}")

# Get files for a specific subject
if subject_ids:
    subject_files = multimodal_dataset.get_subject_files(subject_ids[0])
    print(f"Subject {subject_ids[0]} has:")
    print(f"  {len(subject_files['eeg'])} EEG files")
    print(f"  {len(subject_files['mri'])} MRI files")
    print(f"  {len(subject_files['fmri'])} fMRI files")

# Find matching runs across modalities
multimodal_runs = multimodal_dataset.get_multimodal_runs()
for run_key, run_files in multimodal_runs.items():
    print(f"\nMultimodal run: {run_key}")
    for modality, files in run_files.items():
        if files:
            print(f"  {modality}: {len(files)} files")
            
    # Example: Get events from both EEG and fMRI for the same run
    if run_files['eeg'] and run_files['fmri']:
        eeg_events = multimodal_dataset.get_eeg_events(run_files['eeg'][0])
        fmri_events = multimodal_dataset.get_fmri_events(run_files['fmri'][0])
        
        print(f"  EEG events: {len(eeg_events)} events")
        print(f"  fMRI events: {len(fmri_events)} events")
        
        # Check for matching event types across modalities
        eeg_event_types = set(eeg_events['trial_type']) if not eeg_events.empty else set()
        fmri_event_types = set(fmri_events['trial_type']) if not fmri_events.empty else set()
        common_events = eeg_event_types.intersection(fmri_event_types)
        
        if common_events:
            print(f"  Common event types: {common_events}")
```

## Version Control and Data Management

This repository uses git for version control but **excludes neuroimaging data files** which should be downloaded and managed separately. The `.gitignore` file has been configured to exclude:

- All data directories (`data/`, `raw_data/`, `processed_data/`, etc.)
- Common neuroimaging file formats (`.nii`, `.edf`, `.bdf`, etc.)
- Python build artifacts and virtual environments
- IDE configuration files
- Logs and temporary files
- Potential credential files

### Working with Data

When using this package:

1. Data files are downloaded to the specified data directory (default: `./data`)
2. These directories are not version-controlled to avoid storing large binary files in git
3. DataLad handles versioning of the actual datasets via its own mechanisms
4. Users should not manually commit any neuroimaging data files to this repository

For sharing specific datasets or analysis results, consider using dedicated neuroimaging repositories or data sharing platforms rather than including them in this codebase.

## DataLad Integration

This package now uses DataLad to download datasets from OpenNeuro. DataLad provides several benefits:

1. **Efficient downloads**: Only download the data you need when you need it
2. **Version control**: Easily get specific versions of datasets
3. **Provenance tracking**: Keep track of data origins and modifications
4. **Reproducibility**: Ensure consistent versions across analyses

The downloader works by cloning the dataset from GitHub using DataLad's interface, then retrieving the actual file content as needed. This approach follows OpenNeuro's official recommended method for programmatic access.

## Project Structure

```
neuroloader/
├── __init__.py        # Package initialization
├── base.py            # Base dataset class with common functionality
├── eeg.py             # EEG dataset class
├── mri.py             # MRI and fMRI dataset classes
├── multimodal.py      # Multimodal dataset handling
└── utils.py           # Utility functions
```

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## Requirements

- Python 3.8+
- DataLad (for downloading datasets)
- Dependencies listed in requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details. 