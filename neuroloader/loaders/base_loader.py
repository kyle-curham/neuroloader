"""Base abstract class for neuroimaging datasets"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import os
import pandas as pd
import requests
from tqdm import tqdm
import shutil
import sys
import json

# Import the centralized logging system
from .. import logger
from ..logger import get_logger, LoggerAdapter

# Get package-specific logger
base_logger = get_logger('loaders.base')

class BaseDataset(ABC):
    """Abstract base class for all neuroimaging datasets.
    
    This class defines the interface that all dataset implementations must follow
    and provides common functionality for downloading and processing data.
    """
    
    def __init__(
        self, 
        dataset_id: str,
        data_dir: Optional[Union[str, Path]] = None,
        version: str = "latest",
        is_derivative: Optional[bool] = None
    ):
        """Initialize a neuroimaging dataset.
        
        Args:
            dataset_id: The unique identifier for the dataset on OpenNeuro
            data_dir: Directory where data will be stored (default: ./data)
            version: Dataset version to use (default: "latest")
            is_derivative: Explicitly specify if this is derivative data (preprocessed)
                           If None, will attempt to auto-detect
        """
        self.dataset_id = dataset_id
        self.data_dir = Path(data_dir) if data_dir else Path("./data")
        self.version = version
        self.dataset_dir = self.data_dir / dataset_id
        
        # Create a logger adapter with dataset context
        self.logger = LoggerAdapter(base_logger, {
            'dataset_id': dataset_id,
            'version': version
        })
        
        # Set the derivative flag
        self._is_derivative = is_derivative
        
        # If not explicitly set, auto-detect if this dataset contains derivatives
        if self._is_derivative is None:
            self._is_derivative = self._detect_derivative_data()
            
        if self._is_derivative:
            self.logger.info("Dataset identified as derivative (preprocessed) data")
        else:
            self.logger.info("Dataset identified as raw data")
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Metadata dictionary
        self.metadata: Dict = {}
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
    
    def is_downloaded(self) -> bool:
        """Check if the dataset has been downloaded.
        
        Returns:
            bool: True if the dataset directory exists and contains files
        """
        if not self.dataset_dir.exists():
            return False
            
        # Check if directory has any files (excluding hidden files)
        files = [f for f in self.dataset_dir.rglob('*') if not str(f).startswith('.')]
        
        return len(files) > 0

    def _detect_derivative_data(self) -> bool:
        """Detect if this dataset contains derivative (preprocessed) data.
        
        In BIDS format, derivatives are stored in a 'derivatives' folder.
        There may also be a dataset_description.json file indicating preprocessing.
        
        Returns:
            bool: True if dataset appears to be derivative data, False otherwise
        """
        # Check if the dataset directory exists
        if not self.dataset_dir.exists():
            self.logger.warning(f"Dataset directory {self.dataset_dir} does not exist yet")
            return False
            
        # Check for common indicators of derivative data
        
        # 1. Check if this is a 'derivatives' directory itself
        if "derivatives" in self.dataset_dir.parts:
            return True
            
        # 2. Check for a derivatives directory within the dataset
        derivatives_dir = self.dataset_dir / "derivatives"
        if derivatives_dir.exists() and derivatives_dir.is_dir():
            # If the only content is a derivatives directory, this is likely derivative data
            non_bids_dirs = [d for d in self.dataset_dir.iterdir() 
                             if d.is_dir() and d.name not in ('derivatives', '.git', '.datalad')]
            if not non_bids_dirs:
                return True
        
        # 3. Check dataset_description.json for derivative indicators
        description_file = self.dataset_dir / "dataset_description.json"
        if description_file.exists():
            try:
                with open(description_file, 'r') as f:
                    description = json.load(f)
                    
                # Check for keys indicating derivatives
                if description.get('GeneratedBy'):
                    return True
                if 'pipeline' in str(description).lower():
                    return True
            except Exception as e:
                self.logger.warning(f"Failed to parse dataset_description.json: {str(e)}")
                
        return False
    
    def download_dataset(self, force: bool = False) -> bool:
        """Download the dataset from OpenNeuro using DataLad Python API.
        
        This method uses the DataLad Python API to download the dataset from OpenNeuro's Git repository.
        
        Args:
            force: If True, remove existing dataset directory and re-download
            
        Returns:
            bool: True if download was successful
        """
        # If dataset already exists and force is not specified, return
        if self.dataset_dir.exists() and not force:
            files_count = len(list(self.dataset_dir.rglob('*')))
            if files_count > 0:
                self.logger.info(f"Dataset already exists with {files_count} files. Use force=True to re-download.")
                return True
                
        # Try importing datalad
        try:
            import datalad.api as dl
        except ImportError:
            self.logger.error("DataLad is not installed. Please install it with: pip install datalad")
            return False
            
        # If force is True and directory exists, remove it
        if force and self.dataset_dir.exists():
            self.logger.info(f"Removing existing dataset directory: {self.dataset_dir}")
            shutil.rmtree(self.dataset_dir)
            
        # Construct the GitHub URL for the OpenNeuro dataset
        github_url = f"https://github.com/OpenNeuroDatasets/{self.dataset_id}.git"
        
        try:
            self.logger.info(f"Downloading dataset {self.dataset_id} using DataLad from {github_url}")
            print(f"\nDownloading dataset {self.dataset_id} from OpenNeuro...")
            
            # Basic install without any special configuration
            dl.install(source=github_url, path=str(self.dataset_dir))
            
            # If version is specified and not "latest", check it out
            if self.version != "latest":
                self.logger.info(f"Checking out version: {self.version}")
                import subprocess
                subprocess.run(["git", "-C", str(self.dataset_dir), "checkout", self.version], 
                              check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Get the dataset content
            ds = dl.Dataset(str(self.dataset_dir))
            self.logger.info(f"Downloading dataset content")
            print("\nDownloading dataset content (this may take some time)...")
            
            # Get the dataset content with git-annex showing progress
            try:
                ds.get(".")
                self.logger.info(f"Successfully downloaded dataset {self.dataset_id}")
                print(f"\nDataset download complete.")
            except Exception as e:
                # Specifically catch the "Not Found" config error
                if "config download failed: Not Found" in str(e):
                    self.logger.warning(f"Git config file not found, but continuing with download: {str(e)}")
                    # Continue with download process as this error is often benign
                    self.logger.info(f"Dataset download completed with minor issues")
                    print(f"\nDataset download complete with minor issues.")
                else:
                    # Re-raise other exceptions
                    raise
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download dataset: {str(e)}")
            return False
    
    @abstractmethod
    def get_recording_files(self) -> List[Path]:
        """Get a list of neuroimaging recording files in the dataset.
        
        Returns:
            List[Path]: List of paths to recording files
        """
        pass
    
    @abstractmethod
    def get_events_dataframe(self, recording_file: Union[str, Path]) -> pd.DataFrame:
        """Create a DataFrame containing events for the specified recording.
        
        Args:
            recording_file: Path to the recording file
            
        Returns:
            pd.DataFrame: DataFrame with events information
        """
        pass
    
    def download_file(self, url: str, target_path: Path, chunk_size: int = 8192) -> bool:
        """Download a file from a URL to the specified path.
        
        Args:
            url: URL of the file to download
            target_path: Path where the file should be saved
            chunk_size: Size of chunks for streaming download
            
        Returns:
            bool: True if download was successful
        """
        if target_path.exists():
            self.logger.info(f"File already exists: {target_path}")
            return True
            
        try:
            self.logger.info(f"Downloading from {url} to {target_path}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            # Ensure the directory exists
            os.makedirs(target_path.parent, exist_ok=True)
            
            # Download with progress bar
            with open(target_path, 'wb') as f:
                with tqdm(
                    total=total_size, 
                    unit='B', 
                    unit_scale=True, 
                    desc=target_path.name
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            self.logger.info(f"Successfully downloaded {target_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Download failed: {str(e)}")
            if target_path.exists():
                target_path.unlink()  # Remove partial file
            return False
    
    def validate_dataset_structure(self) -> bool:
        """Validate the structure of the downloaded dataset.
        
        Returns:
            bool: True if the dataset structure is valid
        """
        if not self.dataset_dir.exists():
            self.logger.error(f"Dataset directory does not exist: {self.dataset_dir}")
            return False
            
        # Check for minimal required files/directories
        # This should be implemented by subclasses as needed
        return True
    
    def describe(self) -> Dict:
        """Get a description of the dataset.
        
        Returns:
            Dict: Dictionary containing dataset metadata
        """
        description = {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "data_dir": str(self.dataset_dir),
            "dataset_type": self.__class__.__name__,
            "metadata": self.metadata
        }
        
        # Add simple download status 
        description["download_status"] = "Downloaded" if self.is_downloaded() else "Not downloaded"
        
        # Add subject information if downloaded
        if self.is_downloaded():
            subject_ids = self.get_subject_ids()
            description["subject_count"] = len(subject_ids)
            description["subjects"] = subject_ids
        
        return description

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for the dataset.
        
        Returns:
            Dict[str, Any]: Dictionary of dataset metadata
        """
        raise NotImplementedError("Subclasses must implement get_metadata method")

    def get_modalities(self) -> List[str]:
        """Get a list of imaging modalities in the dataset.
        
        Returns:
            List[str]: List of modalities (e.g., ["T1w", "bold", "eeg"])
        """
        raise NotImplementedError("Subclasses must implement get_modalities method")

    def get_subject_ids(self) -> List[str]:
        """Get a list of all subject IDs in the dataset.
        
        This method extracts subject IDs from the BIDS directory structure, filenames,
        and participants.tsv file. It's a common functionality across all dataset types.
        
        Returns:
            List[str]: List of unique subject IDs
        """
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            self.logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return []
            
        # Set to store unique subject IDs
        subject_ids = set()
        
        # Method 1: Parse participants.tsv file (recommended BIDS way)
        participants_file = self.dataset_dir / 'participants.tsv'
        if participants_file.exists():
            try:
                participants_df = pd.read_csv(participants_file, sep='\t')
                if 'participant_id' in participants_df.columns:
                    # Extract subject IDs from participant_id column (format: sub-XXX)
                    import re
                    sub_pattern = re.compile(r'sub-(\w+)')
                    
                    for participant_id in participants_df['participant_id']:
                        if pd.notna(participant_id):
                            match = sub_pattern.match(participant_id)
                            if match:
                                subject_ids.add(match.group(1))
                            else:
                                # If it doesn't match the pattern, but isn't NaN, add it directly
                                if participant_id and str(participant_id).lower() != 'nan':
                                    subject_ids.add(str(participant_id).replace('sub-', ''))
                self.logger.info(f"Extracted {len(subject_ids)} subject IDs from participants.tsv")
            except Exception as e:
                self.logger.warning(f"Failed to parse participants.tsv: {str(e)}")
        
        # Method 2: Look for subject directories (standard BIDS approach)
        if not subject_ids:
            # Look for directories matching sub-XXX pattern
            import re
            sub_dir_pattern = re.compile(r'sub-(\w+)')
            
            # Find all subject directories in the dataset
            for item in self.dataset_dir.glob('sub-*'):
                if item.is_dir():
                    match = sub_dir_pattern.match(item.name)
                    if match:
                        subject_ids.add(match.group(1))
            
            # Also check for subject directories inside session directories
            for item in self.dataset_dir.glob('ses-*/sub-*'):
                if item.is_dir():
                    match = sub_dir_pattern.match(item.name)
                    if match:
                        subject_ids.add(match.group(1))
                        
            self.logger.info(f"Extracted {len(subject_ids)} subject IDs from directory names")
        
        # Method 3: Look at file names (use existing util function)
        from ..utils import parse_bids_filename, find_files_by_extension
        
        # If we still don't have subjects or want to validate what we found
        # First, we need to get a list of relevant files
        common_extensions = ['.nii', '.nii.gz', '.json', '.tsv', '.set', '.vhdr', '.edf']
        all_files = find_files_by_extension(self.dataset_dir, common_extensions)
        
        # Extract subject IDs from filenames
        for file_path in all_files:
            file_parts = parse_bids_filename(file_path.name)
            if 'sub' in file_parts:
                subject_ids.add(file_parts['sub'])
        
        if not subject_ids:
            self.logger.warning("No subject IDs found in the dataset")
        else:
            self.logger.info(f"Final subject count: {len(subject_ids)}")
            
        return sorted(list(subject_ids))
    
    @abstractmethod
    def get_subject_files(self, subject_id: str) -> List[Path]:
        """Get all files for a specific subject.
        
        This method should be implemented by each modality-specific subclass
        to handle finding files using the appropriate file extensions and patterns.
        
        Args:
            subject_id: Subject ID to get files for
            
        Returns:
            List[Path]: List of paths to files for the subject
        """
        pass

    def __str__(self) -> str:
        """String representation of the dataset.
        
        Returns:
            str: String representation
        """
        return f"{self.__class__.__name__}(dataset_id={self.dataset_id}, version={self.version})"

    def find_files(self, pattern: str) -> List[Path]:
        """Find files matching a glob pattern in the dataset directory.
        
        Args:
            pattern: Glob pattern to match
            
        Returns:
            List[Path]: List of paths to matching files
        """
        # Check if the dataset is downloaded
        if not self.is_downloaded():
            base_logger.warning("Dataset not yet downloaded. Use download_dataset() first.")
            return []
            
        return list(self.dataset_dir.glob(pattern))
