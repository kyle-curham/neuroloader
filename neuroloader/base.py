"""Base abstract class for neuroimaging datasets"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union
import os
import pandas as pd
import requests
from tqdm import tqdm
import logging
import shutil
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseDataset(ABC):
    """Abstract base class for all neuroimaging datasets.
    
    This class defines the interface that all dataset implementations must follow
    and provides common functionality for downloading and processing data.
    """
    
    def __init__(
        self, 
        dataset_id: str,
        data_dir: Optional[Union[str, Path]] = None,
        version: str = "latest"
    ):
        """Initialize a neuroimaging dataset.
        
        Args:
            dataset_id: The unique identifier for the dataset on OpenNeuro
            data_dir: Directory where data will be stored (default: ./data)
            version: Dataset version to use (default: "latest")
        """
        self.dataset_id = dataset_id
        self.data_dir = Path(data_dir) if data_dir else Path("./data")
        self.version = version
        self.dataset_dir = self.data_dir / dataset_id
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Metadata dictionary
        self.metadata: Dict = {}
        
        logger.info(f"Initialized {self.__class__.__name__} with ID: {dataset_id}")
    
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
                logger.info(f"Dataset already exists with {files_count} files. Use force=True to re-download.")
                return True
                
        # Try importing datalad
        try:
            import datalad.api as dl
        except ImportError:
            logger.error("DataLad is not installed. Please install it with: pip install datalad")
            return False
            
        # If force is True and directory exists, remove it
        if force and self.dataset_dir.exists():
            logger.info(f"Removing existing dataset directory: {self.dataset_dir}")
            shutil.rmtree(self.dataset_dir)
            
        # Construct the GitHub URL for the OpenNeuro dataset
        github_url = f"https://github.com/OpenNeuroDatasets/{self.dataset_id}.git"
        
        try:
            logger.info(f"Downloading dataset {self.dataset_id} using DataLad from {github_url}")
            
            # Install the dataset using datalad
            dl.install(source=github_url, path=str(self.dataset_dir))
            
            # If version is specified and not "latest", check it out
            if self.version != "latest":
                logger.info(f"Checking out version: {self.version}")
                import subprocess
                subprocess.run(["git", "-C", str(self.dataset_dir), "checkout", self.version], 
                              check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Get the dataset content
            ds = dl.Dataset(str(self.dataset_dir))
            logger.info(f"Downloading dataset content")
            ds.get(".")
            
            logger.info(f"Successfully downloaded dataset {self.dataset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {str(e)}")
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
            logger.info(f"File already exists: {target_path}")
            return True
            
        try:
            logger.info(f"Downloading from {url} to {target_path}")
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
            
            logger.info(f"Successfully downloaded {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            if target_path.exists():
                target_path.unlink()  # Remove partial file
            return False
    
    def validate_dataset_structure(self) -> bool:
        """Validate the structure of the downloaded dataset.
        
        Returns:
            bool: True if the dataset structure is valid
        """
        if not self.dataset_dir.exists():
            logger.error(f"Dataset directory does not exist: {self.dataset_dir}")
            return False
            
        # Check for minimal required files/directories
        # This should be implemented by subclasses as needed
        return True
    
    def describe(self) -> Dict:
        """Get a description of the dataset.
        
        Returns:
            Dict: Dictionary containing dataset metadata
        """
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "data_dir": str(self.dataset_dir),
            "dataset_type": self.__class__.__name__,
            "metadata": self.metadata
        } 