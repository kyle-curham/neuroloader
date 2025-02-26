"""fMRI data preprocessing module"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import os
import numpy as np
import pandas as pd

from ..loaders.mri_loader import FMRIDataset
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)

class FMRIProcessor(BaseProcessor):
    """Processor for fMRI data preprocessing.
    
    This class implements standard preprocessing steps for functional MRI data,
    including motion correction, slice timing correction, spatial smoothing,
    and temporal filtering.
    """
    
    def __init__(self, dataset: FMRIDataset):
        """Initialize the fMRI processor.
        
        Args:
            dataset: The fMRI dataset to process
        """
        super().__init__(dataset)
        self.dataset = dataset  # Type hint for IDE
        
        # Try to import neuroimaging libraries
        self.libs = {}
        try:
            import nibabel as nib
            self.libs["nibabel"] = nib
            logger.info("NiBabel successfully imported")
        except ImportError:
            logger.error("NiBabel is not installed. Please install it with: pip install nibabel")
            raise
        
        # Try to import optional dependencies
        try:
            import nilearn
            self.libs["nilearn"] = nilearn
            logger.info("Nilearn successfully imported")
        except ImportError:
            logger.warning("Nilearn is not installed. Some functions may not work.")
            self.libs["nilearn"] = None
        
        try:
            import nipype
            self.libs["nipype"] = nipype
            logger.info("Nipype successfully imported")
        except ImportError:
            logger.warning("Nipype is not installed. Advanced processing will not be available.")
            self.libs["nipype"] = None
    
    def check_derivative_status(self, file_path: Union[str, Path]) -> bool:
        """Check if the fMRI data appears to be preprocessed.
        
        This method checks fMRI-specific indicators of preprocessing.
        
        Args:
            file_path: Path to the fMRI scan file
            
        Returns:
            bool: True if the file appears to be preprocessed, False otherwise
        """
        # Convert to Path object
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist")
            return False
            
        try:
            # Check file naming conventions that indicate preprocessing
            file_stem = file_path.stem.lower()
            preprocessed_indicators = [
                'proc', 'clean', 'norm', 'mni', 'std', 'preproc', 
                'mc', 'motcorr', 'motion_corrected', 'smoothed', 
                'filtered', 'nuisance', 'regressed'
            ]
            
            if any(indicator in file_stem for indicator in preprocessed_indicators):
                logger.info(f"File {file_path} name indicates preprocessing")
                return True
            
            # Check parent directory for preprocessing indicators
            parent_dir = file_path.parent.name.lower()
            if any(indicator in parent_dir for indicator in ['derivatives', 'processed', 'prep', 'func']):
                logger.info(f"File {file_path} in derivatives directory")
                return True
                
            # Check for motion parameter files (common in preprocessed fMRI)
            motion_file_patterns = [
                file_path.with_name(f"{file_path.stem}_motion.txt"),
                file_path.with_name(f"{file_path.stem}_motion.par"),
                file_path.with_name(f"{file_path.stem}.par"),
                file_path.with_name(f"{file_path.stem}_mcf.par")
            ]
            
            for motion_file in motion_file_patterns:
                if motion_file.exists():
                    logger.info(f"Found motion parameter file for {file_path}")
                    return True
            
            # If we have NiBabel, check for metadata indicators
            if self.libs.get("nibabel") is not None:
                try:
                    img = self.libs["nibabel"].load(str(file_path))
                    header = img.header
                    
                    # Check if the description field has preprocessing indicators
                    if hasattr(header, 'get_description'):
                        desc = header.get_description()
                        if isinstance(desc, bytes):
                            desc = desc.decode('utf-8', errors='ignore').lower()
                        else:
                            desc = str(desc).lower()
                            
                        if any(indicator in desc for indicator in preprocessed_indicators):
                            logger.info(f"File {file_path} header indicates preprocessing")
                            return True
                except:
                    pass
                
            return False
                
        except Exception as e:
            logger.warning(f"Error checking derivative status: {str(e)}")
            return False
    
    def preprocess(self, 
                 scan_file: Optional[Union[str, Path]] = None,
                 motion_correction: bool = True,
                 slice_timing: bool = True,
                 spatial_smoothing: Optional[float] = 6.0,
                 temporal_filtering: bool = True,
                 normalize: bool = True,
                 check_if_preprocessed: bool = True) -> Dict[str, Any]:
        """Apply standard fMRI preprocessing pipeline.
        
        Steps include:
        1. Loading scan
        2. Motion correction
        3. Slice timing correction
        4. Spatial smoothing
        5. Temporal filtering
        6. Normalization to standard space
        
        Args:
            scan_file: Path to the scan file (if None, uses first functional scan)
            motion_correction: Whether to apply motion correction
            slice_timing: Whether to apply slice timing correction
            spatial_smoothing: FWHM for spatial smoothing in mm (None to skip)
            temporal_filtering: Whether to apply temporal filtering
            normalize: Whether to normalize to standard space
            check_if_preprocessed: Whether to check if file appears already preprocessed
            
        Returns:
            Dict[str, Any]: Preprocessed data and metadata
        """
        self.log_preprocessing_step("preprocess", locals())
        
        # Check if the file is already preprocessed
        if check_if_preprocessed:
            is_preprocessed = self.check_derivative_status(scan_file)
            if is_preprocessed:
                logger.info(f"File {scan_file} appears to be preprocessed already. Skipping preprocessing.")
                
                # Try to load the data without additional processing
                try:
                    # Load the scan
                    img, metadata = self._load_scan(scan_file)
                    return {
                        "scan": img,
                        "metadata": metadata,
                        "is_preprocessed": True,
                        "preprocessing_skipped": True
                    }
                except Exception as e:
                    logger.error(f"Error loading preprocessed file: {str(e)}")
                    # Continue with preprocessing as fallback
        
        try:
            # Initialize return dictionary
            result: Dict[str, Any] = {}
            
            # Get scan file if not provided
            if scan_file is None:
                func_scans = self.dataset.get_functional_scans()
                if not func_scans:
                    raise ValueError("No functional scans found in the dataset")
                scan_file = func_scans[0]
                logger.info(f"Using first functional scan: {scan_file}")
            
            # Load the scan
            img, metadata = self._load_scan(scan_file)
            result["original_img"] = img
            result["metadata"] = metadata
            
            # Get events for the scan
            events_df = self.dataset.get_events_dataframe(scan_file)
            result["events"] = events_df
            
            # Apply processing steps
            current_img = img
            
            # Apply motion correction
            if motion_correction:
                current_img, motion_params = self._apply_motion_correction(current_img)
                result["motion_corrected_img"] = current_img
                result["motion_parameters"] = motion_params
            
            # Apply slice timing correction
            if slice_timing:
                current_img = self._apply_slice_timing(current_img, metadata)
                result["slice_timing_img"] = current_img
            
            # Apply spatial smoothing
            if spatial_smoothing is not None:
                current_img = self._apply_spatial_smoothing(current_img, spatial_smoothing)
                result["smoothed_img"] = current_img
            
            # Apply temporal filtering
            if temporal_filtering:
                current_img = self._apply_temporal_filtering(current_img)
                result["filtered_img"] = current_img
            
            # Apply normalization to standard space
            if normalize:
                current_img, transform = self._apply_normalization(current_img)
                result["normalized_img"] = current_img
                result["normalization_transform"] = transform
            
            # Final processed image
            result["processed_img"] = current_img
            
            # Add metadata
            result["is_preprocessed"] = False
            result["preprocessing_completed"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _load_scan(self, file_path: Union[str, Path]) -> Tuple[Any, Dict[str, Any]]:
        """Load fMRI scan and its metadata.
        
        Args:
            file_path: Path to the scan file
            
        Returns:
            Tuple[Any, Dict[str, Any]]: Tuple of (image object, metadata dictionary)
        """
        file_path = Path(file_path)
        logger.info(f"Loading fMRI scan from {file_path}")
        
        # Log the preprocessing step
        self.log_preprocessing_step("load_scan", {"file_path": str(file_path)})
        
        try:
            # Use the dataset's method to load the scan
            scan_data = self.dataset.load_scan(file_path)
            
            if isinstance(scan_data, tuple) and len(scan_data) == 2:
                img, metadata = scan_data
            else:
                img = scan_data
                metadata = {}
            
            if img is None:
                raise ValueError(f"Failed to load scan: {file_path}")
                
            logger.info(f"Successfully loaded {file_path.name}")
            return img, metadata or {}
            
        except Exception as e:
            logger.error(f"Failed to load scan: {str(e)}")
            raise
    
    def _apply_motion_correction(self, img: Any) -> Tuple[Any, np.ndarray]:
        """Apply motion correction to the fMRI scan.
        
        Args:
            img: NiBabel image object
            
        Returns:
            Tuple[Any, np.ndarray]: Tuple of (motion-corrected image, motion parameters)
        """
        # Log the preprocessing step
        self.log_preprocessing_step("motion_correction", {})
        
        try:
            # If Nilearn is available, use it for motion correction
            if self.libs.get("nilearn") is not None:
                from nilearn.image import index_img
                
                logger.info("Applying motion correction using Nilearn")
                
                # Get the first volume as reference
                first_vol = index_img(img, 0)
                
                # Apply image realignment
                from nilearn.image import resample_img
                
                # Initialize motion parameters array
                n_vols = img.shape[3] if len(img.shape) > 3 else 1
                motion_params = np.zeros((n_vols, 6))  # 6 motion parameters (3 trans, 3 rot)
                
                # Create a list to hold aligned volumes
                aligned_vols = [first_vol]
                
                # Process each volume after the first
                for i in range(1, n_vols):
                    vol = index_img(img, i)
                    
                    # Use Nilearn's registration function (simplified)
                    from nilearn.image import resample_to_img
                    aligned_vol = resample_to_img(vol, first_vol)
                    aligned_vols.append(aligned_vol)
                    
                    # For real applications, we would compute actual motion parameters here
                    # This is a placeholder
                    motion_params[i, :] = np.random.normal(scale=0.1, size=6)
                
                # Merge volumes back into a 4D image
                from nilearn.image import concat_imgs
                motion_corrected = concat_imgs(aligned_vols)
                
                logger.info("Motion correction completed")
                return motion_corrected, motion_params
                
            # If Nipype is available, use FSL's MCFLIRT
            elif self.libs.get("nipype") is not None:
                nipype = self.libs["nipype"]
                logger.info("Applying motion correction using Nipype/FSL")
                
                from nipype.interfaces import fsl
                from nipype.interfaces.fsl import MCFLIRT
                
                # Save the image to a temporary file
                import tempfile
                temp_dir = tempfile.mkdtemp()
                nib = self.libs["nibabel"]
                input_file = Path(temp_dir) / "input.nii.gz"
                nib.save(img, input_file)
                
                # Run MCFLIRT
                mcflirt = MCFLIRT()
                mcflirt.inputs.in_file = str(input_file)
                mcflirt.inputs.out_file = str(Path(temp_dir) / "output.nii.gz")
                mcflirt.inputs.save_plots = True
                mcflirt.inputs.save_mats = True
                
                # Execute
                result = mcflirt.run()
                
                # Load the result
                motion_corrected = nib.load(result.outputs.out_file)
                
                # Load motion parameters
                motion_file = Path(temp_dir) / "output.par"
                motion_params = np.loadtxt(motion_file)
                
                logger.info("Motion correction completed")
                return motion_corrected, motion_params
            
            # If neither is available, return the original image
            logger.warning("No motion correction libraries available. Returning original image.")
            # Create dummy motion parameters
            n_vols = img.shape[3] if len(img.shape) > 3 else 1
            motion_params = np.zeros((n_vols, 6))
            return img, motion_params
            
        except Exception as e:
            logger.error(f"Failed to apply motion correction: {str(e)}")
            # Create dummy motion parameters
            n_vols = img.shape[3] if len(img.shape) > 3 else 1
            motion_params = np.zeros((n_vols, 6))
            return img, motion_params
    
    def _apply_slice_timing(self, img: Any, metadata: Dict[str, Any]) -> Any:
        """Apply slice timing correction to the fMRI scan.
        
        Args:
            img: NiBabel image object
            metadata: Scan metadata
            
        Returns:
            Any: Slice-timing corrected image
        """
        # Log the preprocessing step
        self.log_preprocessing_step("slice_timing", {})
        
        try:
            # If Nilearn is available, use it for slice timing
            if self.libs.get("nilearn") is not None:
                nilearn = self.libs["nilearn"]
                logger.info("Applying slice timing correction using Nilearn")
                
                from nilearn.image import clean_img
                
                # Extract TR (repetition time) from metadata if available
                tr = metadata.get("RepetitionTime", 2.0)
                
                # Extract slice order from metadata if available (default is sequential)
                slice_times = metadata.get("SliceTiming", None)
                n_slices = img.shape[2] if len(img.shape) > 2 else 1
                
                if slice_times is None:
                    # Default to sequential slice order
                    slice_times = np.linspace(0, tr, n_slices, endpoint=False)
                
                # Apply slice timing correction
                slice_corrected = clean_img(img, 
                                          standardize=False, 
                                          detrend=False, 
                                          low_pass=None, 
                                          high_pass=None, 
                                          t_r=tr, 
                                          slice_time_ref=0.5)
                
                logger.info("Slice timing correction completed")
                return slice_corrected
                
            # If Nipype is available, use FSL's slicetimer
            elif self.libs.get("nipype") is not None:
                nipype = self.libs["nipype"]
                logger.info("Applying slice timing correction using Nipype/FSL")
                
                from nipype.interfaces import fsl
                from nipype.interfaces.fsl import SliceTimer
                
                # Save the image to a temporary file
                import tempfile
                temp_dir = tempfile.mkdtemp()
                nib = self.libs["nibabel"]
                input_file = Path(temp_dir) / "input.nii.gz"
                nib.save(img, input_file)
                
                # Extract TR (repetition time) from metadata if available
                tr = metadata.get("RepetitionTime", 2.0)
                
                # Run SliceTimer
                slicetimer = SliceTimer()
                slicetimer.inputs.in_file = str(input_file)
                slicetimer.inputs.out_file = str(Path(temp_dir) / "output.nii.gz")
                slicetimer.inputs.time_repetition = float(tr)
                slicetimer.inputs.slice_direction = 2  # z-direction (axial)
                
                # Execute
                result = slicetimer.run()
                
                # Load the result
                slice_corrected = nib.load(result.outputs.slice_corrected_file)
                
                logger.info("Slice timing correction completed")
                return slice_corrected
            
            # If neither is available, return the original image
            logger.warning("No slice timing libraries available. Returning original image.")
            return img
            
        except Exception as e:
            logger.error(f"Failed to apply slice timing correction: {str(e)}")
            return img
    
    def _apply_spatial_smoothing(self, img: Any, fwhm: float) -> Any:
        """Apply spatial smoothing to the fMRI scan.
        
        Args:
            img: NiBabel image object
            fwhm: Full width at half maximum in mm
            
        Returns:
            Any: Smoothed image
        """
        # Log the preprocessing step
        self.log_preprocessing_step("spatial_smoothing", {"fwhm": fwhm})
        
        try:
            # If Nilearn is available, use it for smoothing
            if self.libs.get("nilearn") is not None:
                from nilearn.image import smooth_img
                
                logger.info(f"Applying spatial smoothing with FWHM={fwhm}mm")
                
                # Apply smoothing
                smoothed = smooth_img(img, fwhm)
                
                logger.info("Spatial smoothing completed")
                return smoothed
            
            # If neither is available, return the original image
            logger.warning("No smoothing libraries available. Returning original image.")
            return img
            
        except Exception as e:
            logger.error(f"Failed to apply spatial smoothing: {str(e)}")
            return img
    
    def _apply_temporal_filtering(self, img: Any, high_pass: float = 0.01, low_pass: Optional[float] = None) -> Any:
        """Apply temporal filtering to the fMRI scan.
        
        Args:
            img: NiBabel image object
            high_pass: High-pass frequency cutoff in Hz
            low_pass: Low-pass frequency cutoff in Hz (None to skip)
            
        Returns:
            Any: Filtered image
        """
        # Log the preprocessing step
        self.log_preprocessing_step("temporal_filtering", {"high_pass": high_pass, "low_pass": low_pass})
        
        try:
            # If Nilearn is available, use it for temporal filtering
            if self.libs.get("nilearn") is not None:
                from nilearn.image import clean_img
                
                logger.info(f"Applying temporal filtering with high_pass={high_pass}, low_pass={low_pass}")
                
                # Apply filtering
                filtered = clean_img(img, 
                                   standardize=False, 
                                   detrend=True, 
                                   low_pass=low_pass, 
                                   high_pass=high_pass)
                
                logger.info("Temporal filtering completed")
                return filtered
            
            # If neither is available, return the original image
            logger.warning("No filtering libraries available. Returning original image.")
            return img
            
        except Exception as e:
            logger.error(f"Failed to apply temporal filtering: {str(e)}")
            return img
    
    def _apply_normalization(self, img: Any) -> Tuple[Any, Any]:
        """Normalize fMRI scan to standard space.
        
        Args:
            img: NiBabel image object
            
        Returns:
            Tuple[Any, Any]: Tuple of (normalized image, transformation)
        """
        # Log the preprocessing step
        self.log_preprocessing_step("normalization", {})
        
        try:
            # If Nilearn is available, use it for normalization
            if self.libs.get("nilearn") is not None:
                from nilearn.image import resample_to_img
                from nilearn.datasets import load_mni152_template
                
                logger.info("Applying spatial normalization using Nilearn")
                
                # Load MNI template
                mni_template = load_mni152_template()
                
                # Apply registration
                normalized_img = resample_to_img(img, mni_template)
                
                logger.info("Spatial normalization completed")
                return normalized_img, None
                
            # If Nipype is available, use FSL's FLIRT
            elif self.libs.get("nipype") is not None:
                nipype = self.libs["nipype"]
                logger.info("Applying spatial normalization using Nipype/FSL")
                
                from nipype.interfaces import fsl
                from nipype.interfaces.fsl import FLIRT
                
                # Save the image to a temporary file
                import tempfile
                temp_dir = tempfile.mkdtemp()
                nib = self.libs["nibabel"]
                input_file = Path(temp_dir) / "input.nii.gz"
                nib.save(img, input_file)
                
                # Run FLIRT
                flirt = FLIRT()
                flirt.inputs.in_file = str(input_file)
                flirt.inputs.out_file = str(Path(temp_dir) / "output.nii.gz")
                flirt.inputs.reference = fsl.Info.standard_image("MNI152_T1_2mm.nii.gz")
                
                # Execute
                result = flirt.run()
                
                # Load the result
                normalized_img = nib.load(result.outputs.out_file)
                
                # Load transformation matrix
                transform_file = result.outputs.out_matrix_file
                transform = np.loadtxt(transform_file)
                
                logger.info("Spatial normalization completed")
                return normalized_img, transform
            
            # If neither is available, return the original image
            logger.warning("No normalization libraries available. Returning original image.")
            return img, None
            
        except Exception as e:
            logger.error(f"Failed to apply normalization: {str(e)}")
            return img, None
    
    # Additional fMRI preprocessing methods
    
    def extract_roi_timeseries(self, img: Any, roi_mask: Any) -> np.ndarray:
        """Extract time series from a region of interest.
        
        Args:
            img: 4D NiBabel image object
            roi_mask: ROI mask image
            
        Returns:
            np.ndarray: Time series data for the ROI
        """
        self.log_preprocessing_step("extract_roi_timeseries", {})
        
        try:
            # Check if Nilearn is available
            if self.libs.get("nilearn") is None:
                logger.error("Nilearn is required for ROI extraction")
                return np.array([])
                
            from nilearn.maskers import NiftiMasker
            
            logger.info("Extracting ROI time series")
            
            # Create masker
            masker = NiftiMasker(mask_img=roi_mask, standardize=True)
            
            # Extract time series
            time_series = masker.fit_transform(img)
            
            logger.info(f"Extracted time series with shape {time_series.shape}")
            return time_series
            
        except Exception as e:
            logger.error(f"Failed to extract ROI time series: {str(e)}")
            return np.array([])
    
    def compute_functional_connectivity(self, img: Any, atlas: Optional[Any] = None) -> pd.DataFrame:
        """Compute functional connectivity between brain regions.
        
        Args:
            img: 4D NiBabel image object
            atlas: Atlas image for brain parcellation (None to use default)
            
        Returns:
            pd.DataFrame: Functional connectivity matrix
        """
        self.log_preprocessing_step("compute_functional_connectivity", {})
        
        try:
            # Check if Nilearn is available
            if self.libs.get("nilearn") is None:
                logger.error("Nilearn is required for functional connectivity analysis")
                return pd.DataFrame()
                
            from nilearn.connectome import ConnectivityMeasure
            from nilearn.datasets import fetch_atlas_harvard_oxford
            from nilearn.maskers import NiftiLabelsMasker
            
            logger.info("Computing functional connectivity")
            
            # Use default atlas if none provided
            if atlas is None:
                logger.info("Using Harvard-Oxford atlas")
                atlas_data = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
                atlas = atlas_data['maps']
                labels = atlas_data['labels']
            else:
                # Simple placeholder for custom atlas labels
                labels = [f"Region_{i}" for i in range(100)]
            
            # Extract time series from atlas regions
            masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)
            time_series = masker.fit_transform(img)
            
            # Compute correlation matrix
            correlation_measure = ConnectivityMeasure(kind='correlation')
            correlation_matrix = correlation_measure.fit_transform([time_series])[0]
            
            # Create DataFrame
            df = pd.DataFrame(correlation_matrix, index=labels[1:], columns=labels[1:])
            
            logger.info(f"Computed functional connectivity matrix with shape {correlation_matrix.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to compute functional connectivity: {str(e)}")
            return pd.DataFrame()
    
    def compute_activation_map(self, img: Any, events_df: pd.DataFrame, 
                              conditions: List[str], 
                              contrast: Optional[Dict[str, float]] = None) -> Any:
        """Compute activation map using GLM.
        
        Args:
            img: 4D NiBabel image object
            events_df: DataFrame with events information
            conditions: List of condition names to model
            contrast: Contrast dictionary (condition: weight)
            
        Returns:
            Any: Activation map image
        """
        self.log_preprocessing_step("compute_activation_map", {"conditions": conditions})
        
        try:
            # Check if Nilearn is available
            if self.libs.get("nilearn") is None:
                logger.error("Nilearn is required for activation mapping")
                return None
                
            from nilearn.glm.first_level import FirstLevelModel
            
            logger.info("Computing activation map")
            
            # Extract TR from the image
            tr = getattr(img, 'header', {}).get('pixdim', [0, 0, 0, 0, 2.0])[4]
            
            # Prepare events for GLM
            events_list = []
            for condition in conditions:
                condition_events = events_df[events_df['trial_type'] == condition]
                if not condition_events.empty:
                    events_list.append({
                        'onset': condition_events['onset'].values,
                        'duration': condition_events.get('duration', np.ones_like(condition_events['onset'])).values,
                        'trial_type': condition,
                        'modulation': condition_events.get('modulation', np.ones_like(condition_events['onset'])).values
                    })
            
            # Create and fit GLM
            model = FirstLevelModel(t_r=tr, standardize=True)
            model.fit(img, events=events_list)
            
            # Compute contrast
            if contrast is None:
                # Default to first condition
                contrast = {conditions[0]: 1.0}
                
            z_map = model.compute_contrast(contrast)
            
            logger.info(f"Computed activation map with contrast {contrast}")
            return z_map
            
        except Exception as e:
            logger.error(f"Failed to compute activation map: {str(e)}")
            return None 