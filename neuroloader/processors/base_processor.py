"""Base abstract class for neuroimaging processors"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..loaders.base_loader import BaseDataset
from ..logger import get_logger

# Get logger for this module
logger = get_logger("processors.base")

class BaseProcessor(ABC):
    """Base class for neuroimaging preprocessing pipelines.
    
    This class defines the interface for preprocessing different modalities
    of neuroimaging data and tracking preprocessing steps.
    """
    
    def __init__(self, dataset: BaseDataset):
        """Initialize the processor with a dataset.
        
        Args:
            dataset: The dataset to process
        """
        self.dataset = dataset
        self.preprocessing_history: List[Dict[str, Any]] = []
        logger.info(f"Initialized {self.__class__.__name__} for dataset: {dataset.dataset_id}")
        
    @abstractmethod
    def preprocess(self, **kwargs) -> Dict[str, Any]:
        """Apply preprocessing steps to the dataset.
        
        This method should implement the standard preprocessing pipeline
        for the specific neuroimaging modality.
        
        Args:
            **kwargs: Additional keyword arguments for preprocessing
            
        Returns:
            Dict[str, Any]: Results and metadata from preprocessing
        """
        pass
        
    def log_preprocessing_step(self, step_name: str, params: Dict[str, Any]) -> None:
        """Log preprocessing steps for reproducibility.
        
        Args:
            step_name: Name of the preprocessing step
            params: Parameters used for this step
        """
        step_info = {
            "step": step_name,
            "params": params,
            "timestamp": datetime.now().isoformat()
        }
        self.preprocessing_history.append(step_info)
        logger.info(f"Applied preprocessing step: {step_name} with params: {params}")
        
    def get_preprocessing_history(self) -> List[Dict[str, Any]]:
        """Get the preprocessing history.
        
        Returns:
            List[Dict[str, Any]]: List of preprocessing steps that have been applied
        """
        return self.preprocessing_history
        
    def clear_preprocessing_history(self) -> None:
        """Clear the preprocessing history."""
        self.preprocessing_history = []
        logger.info("Cleared preprocessing history")
        
    def save_preprocessing_history(self, output_file: str) -> None:
        """Save the preprocessing history to a JSON file.
        
        Args:
            output_file: Path to the output file
        """
        import json
        with open(output_file, 'w') as f:
            json.dump(self.preprocessing_history, f, indent=2)
        logger.info(f"Saved preprocessing history to {output_file}") 