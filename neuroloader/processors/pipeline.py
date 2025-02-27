"""Pipeline for chaining neuroimaging preprocessing steps"""

import os
import json
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from pathlib import Path
from datetime import datetime

from .base_processor import BaseProcessor
from ..loaders.base_loader import BaseDataset
from ..logger import get_logger

# Get logger for this module
logger = get_logger("processors.pipeline")

class PreprocessingPipeline:
    """Pipeline for chaining multiple preprocessing steps.
    
    This class allows for creating complex preprocessing workflows by
    combining multiple processors and custom functions in a sequential
    or branched pipeline.
    """
    
    def __init__(self, name: str = "preprocessing_pipeline", skip_if_derivative: bool = True):
        """Initialize the preprocessing pipeline.
        
        Args:
            name: Name of the pipeline for identification
            skip_if_derivative: Whether to skip preprocessing for derivative datasets
        """
        self.name = name
        self.skip_if_derivative = skip_if_derivative
        self.steps: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        logger.info(f"Initialized preprocessing pipeline: {name}")
    
    def add_processor(self, 
                     processor: BaseProcessor, 
                     name: str,
                     params: Optional[Dict[str, Any]] = None,
                     input_keys: Optional[Dict[str, str]] = None,
                     output_keys: Optional[List[str]] = None,
                     force_execution: bool = False) -> 'PreprocessingPipeline':
        """Add a processor to the pipeline.
        
        Args:
            processor: The neuroimaging processor to add
            name: Name of this step for identification
            params: Parameters to pass to the processor's preprocess method
            input_keys: Mapping of processor parameter names to pipeline result keys
            output_keys: List of keys to extract from processor output to pipeline results
            force_execution: Whether to execute this step even for derivative datasets
            
        Returns:
            self: For method chaining
        """
        step = {
            "type": "processor",
            "processor": processor,
            "name": name,
            "params": params or {},
            "input_keys": input_keys or {},
            "output_keys": output_keys or [],
            "force_execution": force_execution
        }
        self.steps.append(step)
        logger.info(f"Added processor step: {name}")
        return self
    
    def add_function(self, 
                    function: Callable, 
                    name: str,
                    params: Optional[Dict[str, Any]] = None,
                    input_keys: Optional[Dict[str, str]] = None,
                    output_key: Optional[str] = None,
                    force_execution: bool = False) -> 'PreprocessingPipeline':
        """Add a custom function to the pipeline.
        
        Args:
            function: The function to add
            name: Name of this step for identification
            params: Fixed parameters to pass to the function
            input_keys: Mapping of function parameter names to pipeline result keys
            output_key: Key to store function output in pipeline results
            force_execution: Whether to execute this step even for derivative datasets
            
        Returns:
            self: For method chaining
        """
        step = {
            "type": "function",
            "function": function,
            "name": name,
            "params": params or {},
            "input_keys": input_keys or {},
            "output_key": output_key,
            "force_execution": force_execution
        }
        self.steps.append(step)
        logger.info(f"Added function step: {name}")
        return self
    
    def add_conditional(self, 
                       condition: Callable[[Dict[str, Any]], bool], 
                       name: str,
                       if_true: List[Dict[str, Any]],
                       if_false: Optional[List[Dict[str, Any]]] = None,
                       force_execution: bool = False) -> 'PreprocessingPipeline':
        """Add a conditional branch to the pipeline.
        
        Args:
            condition: Function that takes the results dict and returns a boolean
            name: Name of this step for identification
            if_true: Steps to execute if condition is True
            if_false: Steps to execute if condition is False
            force_execution: Whether to execute this step even for derivative datasets
            
        Returns:
            self: For method chaining
        """
        step = {
            "type": "conditional",
            "condition": condition,
            "name": name,
            "if_true": if_true,
            "if_false": if_false or [],
            "force_execution": force_execution
        }
        self.steps.append(step)
        logger.info(f"Added conditional step: {name}")
        return self
    
    def check_derivative_status(self, datasets: List[BaseDataset]) -> bool:
        """Check if any of the datasets used in the pipeline are derivative.
        
        Args:
            datasets: List of datasets used in this pipeline
            
        Returns:
            bool: True if any dataset is derivative, False otherwise
        """
        for dataset in datasets:
            if dataset.is_derivative():
                logger.info(f"Dataset {dataset.dataset_id} is a derivative dataset")
                return True
        return False
    
    def execute(self, 
               initial_results: Optional[Dict[str, Any]] = None,
               datasets: Optional[List[BaseDataset]] = None) -> Dict[str, Any]:
        """Execute the preprocessing pipeline.
        
        Args:
            initial_results: Initial data to start with
            datasets: List of datasets used in this pipeline, used to check derivative status
            
        Returns:
            Dict[str, Any]: Results from all processing steps
        """
        # Initialize or update results
        if initial_results is not None:
            self.results = initial_results.copy()
        else:
            self.results = {}
        
        # Clear execution history
        self.execution_history = []
        
        # Start time for the entire pipeline
        pipeline_start_time = datetime.now()
        logger.info(f"Executing pipeline: {self.name}")
        
        # Check if we're working with derivative data
        should_skip_preprocessing = False
        if self.skip_if_derivative and datasets:
            should_skip_preprocessing = self.check_derivative_status(datasets)
            if should_skip_preprocessing:
                logger.info("Derivative dataset detected - skipping preprocessing steps unless explicitly forced")
                self.results["preprocessing_skipped"] = True
        
        # Process each step
        for step_idx, step in enumerate(self.steps):
            step_start_time = datetime.now()
            step_type = step["type"]
            step_name = step["name"]
            force_execution = step.get("force_execution", False)
            
            # Skip preprocessing steps for derivative data unless forced
            if should_skip_preprocessing and not force_execution:
                logger.info(f"Skipping step {step_idx+1}/{len(self.steps)}: {step_name} (derivative dataset)")
                
                # Record skipped execution
                step_end_time = datetime.now()
                execution_record = {
                    "step_idx": step_idx,
                    "step_name": step_name,
                    "step_type": step_type,
                    "start_time": step_start_time.isoformat(),
                    "end_time": step_end_time.isoformat(),
                    "duration_seconds": 0,
                    "status": "skipped",
                    "reason": "derivative_dataset"
                }
                self.execution_history.append(execution_record)
                continue
            
            logger.info(f"Executing step {step_idx+1}/{len(self.steps)}: {step_name} ({step_type})")
            
            try:
                if step_type == "processor":
                    self._execute_processor_step(step)
                elif step_type == "function":
                    self._execute_function_step(step)
                elif step_type == "conditional":
                    self._execute_conditional_step(step)
                else:
                    logger.warning(f"Unknown step type: {step_type}")
                    
                # Record execution details
                step_end_time = datetime.now()
                execution_record = {
                    "step_idx": step_idx,
                    "step_name": step_name,
                    "step_type": step_type,
                    "start_time": step_start_time.isoformat(),
                    "end_time": step_end_time.isoformat(),
                    "duration_seconds": (step_end_time - step_start_time).total_seconds(),
                    "status": "success"
                }
                self.execution_history.append(execution_record)
                
            except Exception as e:
                logger.error(f"Error executing step {step_name}: {str(e)}")
                
                # Record execution error
                step_end_time = datetime.now()
                execution_record = {
                    "step_idx": step_idx,
                    "step_name": step_name,
                    "step_type": step_type,
                    "start_time": step_start_time.isoformat(),
                    "end_time": step_end_time.isoformat(),
                    "duration_seconds": (step_end_time - step_start_time).total_seconds(),
                    "status": "error",
                    "error": str(e)
                }
                self.execution_history.append(execution_record)
                
                # Depending on configuration, we might want to raise the exception
                # or continue with the next step
                raise
        
        # Calculate total execution time
        pipeline_end_time = datetime.now()
        total_duration = (pipeline_end_time - pipeline_start_time).total_seconds()
        logger.info(f"Pipeline execution completed in {total_duration:.2f} seconds")
        
        return self.results
    
    def _execute_processor_step(self, step: Dict[str, Any]) -> None:
        """Execute a processor step.
        
        Args:
            step: Step configuration dictionary
        """
        processor = step["processor"]
        params = step["params"].copy()
        
        # Map inputs from results to parameters
        for param_name, result_key in step["input_keys"].items():
            if result_key in self.results:
                params[param_name] = self.results[result_key]
            else:
                logger.warning(f"Result key '{result_key}' not found for parameter '{param_name}'")
        
        # Execute the processor
        processor_result = processor.preprocess(**params)
        
        # Store specific outputs in results
        output_keys = step["output_keys"]
        if output_keys:
            for key in output_keys:
                if key in processor_result:
                    self.results[key] = processor_result[key]
                else:
                    logger.warning(f"Output key '{key}' not found in processor result")
        else:
            # If no output_keys specified, store all results under the step name
            self.results[step["name"]] = processor_result
    
    def _execute_function_step(self, step: Dict[str, Any]) -> None:
        """Execute a function step.
        
        Args:
            step: Step configuration dictionary
        """
        function = step["function"]
        params = step["params"].copy()
        
        # Map inputs from results to parameters
        for param_name, result_key in step["input_keys"].items():
            if result_key in self.results:
                params[param_name] = self.results[result_key]
            else:
                logger.warning(f"Result key '{result_key}' not found for parameter '{param_name}'")
        
        # Execute the function
        function_result = function(**params)
        
        # Store the output
        output_key = step["output_key"]
        if output_key:
            self.results[output_key] = function_result
        else:
            # If no output_key specified, store under the step name
            self.results[step["name"]] = function_result
    
    def _execute_conditional_step(self, step: Dict[str, Any]) -> None:
        """Execute a conditional step.
        
        Args:
            step: Step configuration dictionary
        """
        condition = step["condition"]
        
        # Evaluate the condition
        condition_result = condition(self.results)
        
        # Record the condition result
        self.results[f"{step['name']}_condition"] = condition_result
        
        # Execute the appropriate branch
        if condition_result:
            steps_to_execute = step["if_true"]
            logger.info(f"Condition '{step['name']}' is True, executing 'if_true' branch")
        else:
            steps_to_execute = step["if_false"]
            logger.info(f"Condition '{step['name']}' is False, executing 'if_false' branch")
        
        # Create a sub-pipeline with the branch steps
        sub_pipeline = PreprocessingPipeline(f"{step['name']}_branch")
        sub_pipeline.steps = steps_to_execute
        
        # Execute the sub-pipeline with the current results
        branch_results = sub_pipeline.execute(self.results)
        
        # Update the main results with the branch results
        self.results.update(branch_results)
    
    def save_pipeline(self, file_path: Union[str, Path]) -> None:
        """Save the pipeline configuration to a JSON file.
        
        Args:
            file_path: Path to save the pipeline configuration
        """
        file_path = Path(file_path)
        
        # Create a serializable version of the pipeline
        serializable_steps = []
        for step in self.steps:
            serializable_step = {
                "type": step["type"],
                "name": step["name"]
            }
            
            if step["type"] == "processor":
                # Store processor class name and parameters
                serializable_step["processor_class"] = step["processor"].__class__.__name__
                serializable_step["params"] = step["params"]
                serializable_step["input_keys"] = step["input_keys"]
                serializable_step["output_keys"] = step["output_keys"]
            elif step["type"] == "function":
                # Store function name and parameters
                serializable_step["function_name"] = step["function"].__name__
                serializable_step["params"] = step["params"]
                serializable_step["input_keys"] = step["input_keys"]
                serializable_step["output_key"] = step["output_key"]
            
            serializable_steps.append(serializable_step)
        
        pipeline_config = {
            "name": self.name,
            "steps": serializable_steps,
            "created_at": datetime.now().isoformat()
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(pipeline_config, f, indent=2)
            
        logger.info(f"Saved pipeline configuration to {file_path}")
    
    def save_results(self, file_path: Union[str, Path]) -> None:
        """Save the pipeline results to a JSON file.
        
        Args:
            file_path: Path to save the results
        """
        file_path = Path(file_path)
        
        # Filter results to include only serializable objects
        serializable_results = {}
        for key, value in self.results.items():
            try:
                # Test JSON serialization
                json.dumps(value)
                serializable_results[key] = value
            except (TypeError, OverflowError):
                logger.warning(f"Result '{key}' is not JSON serializable, skipping")
                # Store information about the skipped result
                serializable_results[f"{key}_info"] = f"Type: {type(value).__name__}"
        
        # Add execution history
        result_data = {
            "pipeline_name": self.name,
            "execution_time": datetime.now().isoformat(),
            "execution_history": self.execution_history,
            "results": serializable_results
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(result_data, f, indent=2)
            
        logger.info(f"Saved pipeline results to {file_path}")
    
    def get_preprocessing_history(self) -> List[Dict[str, Any]]:
        """Get the combined preprocessing history from all processors.
        
        Returns:
            List[Dict[str, Any]]: Combined preprocessing history
        """
        combined_history = []
        
        # Get all processor instances from the pipeline
        processors = []
        for step in self.steps:
            if step["type"] == "processor":
                processors.append(step["processor"])
        
        # Get preprocessing history from each processor
        for processor in processors:
            history = processor.get_preprocessing_history()
            for item in history:
                # Add processor information to each history item
                item_with_processor = item.copy()
                item_with_processor["processor"] = processor.__class__.__name__
                combined_history.append(item_with_processor)
        
        # Sort by timestamp
        combined_history.sort(key=lambda x: x.get("timestamp", ""))
        
        return combined_history
    
    def clear_history(self) -> None:
        """Clear preprocessing history from all processors and pipeline execution history."""
        # Clear each processor's history
        for step in self.steps:
            if step["type"] == "processor":
                step["processor"].clear_preprocessing_history()
        
        # Clear pipeline execution history
        self.execution_history = []
        logger.info("Cleared all preprocessing history") 