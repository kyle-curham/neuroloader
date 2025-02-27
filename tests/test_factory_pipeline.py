"""Tests for the factory module's preprocessing pipeline generation functionality."""

import os
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from neuroloader.factory import create_dataset, create_preprocessing_pipeline
from neuroloader.loaders.base_loader import BaseDataset
from neuroloader.loaders.eeg_loader import EEGDataset
from neuroloader.loaders.mri_loader import MRIDataset, FMRIDataset
from neuroloader.loaders.multimodal_loader import MultimodalDataset
from neuroloader.processors.pipeline import PreprocessingPipeline


class TestFactoryPipeline(unittest.TestCase):
    """Test the factory module's pipeline generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock datasets for testing
        self.mock_eeg_dataset = MagicMock(spec=EEGDataset)
        self.mock_eeg_dataset.dataset_id = "test_eeg"
        self.mock_eeg_dataset.is_derivative.return_value = False
        
        self.mock_mri_dataset = MagicMock(spec=MRIDataset)
        self.mock_mri_dataset.dataset_id = "test_mri"
        self.mock_mri_dataset.is_derivative.return_value = False
        
        self.mock_fmri_dataset = MagicMock(spec=FMRIDataset)
        self.mock_fmri_dataset.dataset_id = "test_fmri"
        self.mock_fmri_dataset.is_derivative.return_value = False
        
        self.mock_multimodal_dataset = MagicMock(spec=MultimodalDataset)
        self.mock_multimodal_dataset.dataset_id = "test_multi"
        self.mock_multimodal_dataset.is_derivative.return_value = False
        self.mock_multimodal_dataset.get_modalities.return_value = ["eeg", "mri", "fmri"]
        
        # Mock the get_modality_dataset method
        def get_modality_dataset_side_effect(modality):
            if modality == "eeg":
                return self.mock_eeg_dataset
            elif modality == "mri":
                return self.mock_mri_dataset
            elif modality == "fmri":
                return self.mock_fmri_dataset
            return None
            
        self.mock_multimodal_dataset.get_modality_dataset.side_effect = get_modality_dataset_side_effect
        
        # Create a derivative dataset
        self.mock_derivative_dataset = MagicMock(spec=MRIDataset)
        self.mock_derivative_dataset.dataset_id = "test_derivative"
        self.mock_derivative_dataset.is_derivative.return_value = True
    
    @patch('neuroloader.processors.eeg_processor.EEGProcessor')
    def test_create_eeg_pipeline(self, mock_eeg_processor_class):
        """Test creating a pipeline for EEG data."""
        # Setup the mock
        mock_eeg_processor = MagicMock()
        mock_eeg_processor_class.return_value = mock_eeg_processor
        
        # Test pipeline creation
        pipeline = create_preprocessing_pipeline(self.mock_eeg_dataset)
        
        # Assert a pipeline was created
        self.assertIsNotNone(pipeline)
        self.assertIsInstance(pipeline, PreprocessingPipeline)
        
        # Assert the pipeline has steps
        self.assertTrue(len(pipeline.steps) > 0)
        
        # Assert EEG processor was created
        mock_eeg_processor_class.assert_called_once()
    
    @patch('neuroloader.processors.mri_processor.MRIProcessor')
    def test_create_mri_pipeline(self, mock_mri_processor_class):
        """Test creating a pipeline for MRI data."""
        # Setup the mock
        mock_mri_processor = MagicMock()
        mock_mri_processor_class.return_value = mock_mri_processor
        
        # Test pipeline creation
        pipeline = create_preprocessing_pipeline(self.mock_mri_dataset)
        
        # Assert a pipeline was created
        self.assertIsNotNone(pipeline)
        self.assertIsInstance(pipeline, PreprocessingPipeline)
        
        # Assert the pipeline has steps
        self.assertTrue(len(pipeline.steps) > 0)
        
        # Assert MRI processor was created
        mock_mri_processor_class.assert_called_once()
    
    @patch('neuroloader.processors.fmri_processor.FMRIProcessor')
    def test_create_fmri_pipeline(self, mock_fmri_processor_class):
        """Test creating a pipeline for fMRI data."""
        # Setup the mock
        mock_fmri_processor = MagicMock()
        mock_fmri_processor_class.return_value = mock_fmri_processor
        
        # Test pipeline creation
        pipeline = create_preprocessing_pipeline(self.mock_fmri_dataset)
        
        # Assert a pipeline was created
        self.assertIsNotNone(pipeline)
        self.assertIsInstance(pipeline, PreprocessingPipeline)
        
        # Assert the pipeline has steps
        self.assertTrue(len(pipeline.steps) > 0)
        
        # Assert fMRI processor was created
        mock_fmri_processor_class.assert_called_once()
    
    @patch('neuroloader.processors.mri_processor.MRIProcessor')
    @patch('neuroloader.processors.fmri_processor.FMRIProcessor')
    @patch('neuroloader.processors.eeg_processor.EEGProcessor')
    def test_create_multimodal_pipeline(self, mock_eeg_processor_class, mock_fmri_processor_class, mock_mri_processor_class):
        """Test creating a pipeline for multimodal data."""
        # Setup the mocks
        mock_eeg_processor = MagicMock()
        mock_eeg_processor_class.return_value = mock_eeg_processor
        
        mock_mri_processor = MagicMock()
        mock_mri_processor_class.return_value = mock_mri_processor
        
        mock_fmri_processor = MagicMock()
        mock_fmri_processor_class.return_value = mock_fmri_processor
        
        # Test pipeline creation
        pipeline = create_preprocessing_pipeline(self.mock_multimodal_dataset)
        
        # Assert a pipeline was created
        self.assertIsNotNone(pipeline)
        self.assertIsInstance(pipeline, PreprocessingPipeline)
        
        # Assert the pipeline has steps
        self.assertTrue(len(pipeline.steps) > 0)
        
        # Assert all processors were created
        mock_eeg_processor_class.assert_called_once()
        mock_mri_processor_class.assert_called_once()
        mock_fmri_processor_class.assert_called_once()
    
    def test_pipeline_respects_derivative_status(self):
        """Test that the pipeline respects derivative status."""
        # Create a pipeline for a derivative dataset
        pipeline = create_preprocessing_pipeline(self.mock_derivative_dataset)
        
        # Assert a pipeline was created
        self.assertIsNotNone(pipeline)
        self.assertIsInstance(pipeline, PreprocessingPipeline)
        
        # Execute the pipeline with our derivative dataset
        results = pipeline.execute(datasets=[self.mock_derivative_dataset])
        
        # Assert preprocessing was skipped
        self.assertTrue(results.get("preprocessing_skipped", False))
    
    @patch('neuroloader.factory.detect_modality_from_filelist')
    @patch('neuroloader.processors.mri_processor.MRIProcessor')
    def test_create_dataset_with_pipeline(self, mock_mri_processor_class, mock_detect_modality):
        """Test creating a dataset with pipeline using the factory."""
        # Setup the mocks
        mock_detect_modality.return_value = {"eeg": False, "mri": True, "fmri": False}
        
        mock_mri_processor = MagicMock()
        mock_mri_processor_class.return_value = mock_mri_processor
        
        # Use patching to avoid actual dataset creation
        with patch('neuroloader.factory.MRIDataset') as mock_mri_dataset_class:
            mock_mri_dataset = MagicMock()
            mock_mri_dataset.dataset_id = "test_dataset"
            mock_mri_dataset.is_derivative.return_value = False
            mock_mri_dataset_class.return_value = mock_mri_dataset
            
            # Test creating a dataset with pipeline
            result = create_dataset(
                "test_dataset",
                data_dir="./data",
                with_pipeline=True
            )
            
            # Assert we got a dictionary with dataset and pipeline
            self.assertIsInstance(result, dict)
            self.assertIn("dataset", result)
            self.assertIn("pipeline", result)
            
            # Assert both the dataset and pipeline were created
            self.assertIsNotNone(result["dataset"])
            self.assertIsNotNone(result["pipeline"])
            
            # Verify the pipeline is for MRI data
            pipeline = result["pipeline"]
            self.assertIsInstance(pipeline, PreprocessingPipeline)


if __name__ == '__main__':
    unittest.main() 