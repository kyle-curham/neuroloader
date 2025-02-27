"""Utility functions for neuroimaging data processing"""

import json
import os
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import hashlib
from tqdm import tqdm
import pandas as pd
import numpy as np

# Import the package's logger
from . import logger

# Use the package's centralized logger
utils_logger = logger.get_logger('utils')

def download_file(url: str, target_path: Union[str, Path], 
                 chunk_size: int = 8192, force: bool = False) -> bool:
    """Download a file from a URL with progress tracking.
    
    Args:
        url: URL of the file to download
        target_path: Path where the file should be saved
        chunk_size: Size of chunks for streaming download
        force: If True, re-download even if the file exists
        
    Returns:
        bool: True if download was successful
    """
    target_path = Path(target_path)
    
    if target_path.exists() and not force:
        utils_logger.info(f"File already exists: {target_path}")
        return True
    
    try:
        # Ensure the directory exists
        os.makedirs(target_path.parent, exist_ok=True)
        
        utils_logger.info(f"Downloading from {url} to {target_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
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
        
        utils_logger.info(f"Successfully downloaded {target_path}")
        return True
        
    except Exception as e:
        utils_logger.error(f"Download failed: {str(e)}")
        if target_path.exists():
            target_path.unlink()  # Remove partial file
        return False

def validate_dataset(dataset_path: Union[str, Path], 
                    required_files: Optional[List[str]] = None) -> bool:
    """Validate that a dataset exists and has required files.
    
    Args:
        dataset_path: Path to the dataset directory
        required_files: List of required files (relative to dataset_path)
        
    Returns:
        bool: True if validation passes
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        utils_logger.error(f"Dataset path does not exist: {dataset_path}")
        return False
    
    if not dataset_path.is_dir():
        utils_logger.error(f"Dataset path is not a directory: {dataset_path}")
        return False
    
    if required_files:
        for file_path in required_files:
            full_path = dataset_path / file_path
            if not full_path.exists():
                utils_logger.error(f"Required file missing: {full_path}")
                return False
    
    return True

def get_file_checksum(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """Calculate checksum of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use
        
    Returns:
        str: Hexadecimal digest of the file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()

def find_files_by_extension(
    root_dir: Union[str, Path], 
    extensions: List[str],
    recursive: bool = True
) -> List[Path]:
    """Find all files with specified extensions.
    
    Args:
        root_dir: Directory to search in
        extensions: List of file extensions to find (e.g. ['.nii', '.nii.gz'])
        recursive: Whether to search recursively
        
    Returns:
        List[Path]: List of found file paths
    """
    root_dir = Path(root_dir)
    
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    # Normalize extensions to include the dot and lowercase
    normalized_extensions = [
        ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
        for ext in extensions
    ]
    
    results = []
    
    if recursive:
        for path in root_dir.rglob('*'):
            if path.is_file() and path.suffix.lower() in normalized_extensions:
                results.append(path)
    else:
        for path in root_dir.glob('*'):
            if path.is_file() and path.suffix.lower() in normalized_extensions:
                results.append(path)
    
    return results

def load_json_file(file_path: Union[str, Path]) -> Dict:
    """Load a JSON file into a dictionary.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dict: Loaded JSON data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_bids_filename(filename: str) -> Dict[str, str]:
    """Parse a BIDS-formatted filename into its components.
    
    Args:
        filename: BIDS filename to parse
        
    Returns:
        Dict[str, str]: Dictionary of BIDS entities
    """
    # Extract the base filename without directory
    base_filename = os.path.basename(filename)
    
    # BIDS entities are in the format key-value_
    entity_pattern = r'([a-zA-Z]+)-([^_]+)'
    
    # Find all entities in the filename
    entities = re.findall(entity_pattern, base_filename)
    
    # Convert to dictionary
    result = {}
    for key, value in entities:
        result[key] = value
    
    # Add file extension
    if '.' in base_filename:
        extension = '.'.join(base_filename.split('.')[-2:]) if base_filename.endswith('.nii.gz') else base_filename.split('.')[-1]
        result['extension'] = extension
    
    return result 