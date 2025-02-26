"""MRI data preprocessing module"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import os
import numpy as np

from ..loaders.mri_loader import MRIDataset
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)

class MRIProcessor(BaseProcessor):
    """Processor for structural MRI data preprocessing.
    
    This class implements standard preprocessing steps for structural MRI data,
    including bias field correction, skull stripping, segmentation, and
    spatial normalization.
    """
    
    def __init__(self, dataset: MRIDataset):
        """Initialize the MRI processor.
        
        Args:
            dataset: The MRI dataset to process
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
            import ants
            self.libs["ants"] = ants
            logger.info("ANTsPy successfully imported")
        except ImportError:
            logger.warning("ANTsPy is not installed. Advanced registration will not be available.")
            self.libs["ants"] = None
    
    def check_derivative_status(self, file_path: Union[str, Path]) -> bool:
        """Check if the MRI data appears to be preprocessed.
        
        This method checks MRI-specific indicators of preprocessing.
        
        Args:
            file_path: Path to the MRI scan file
            
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
                'skullstrip', 'brain_extracted', 'bias_corrected', 'reg',
                'segmented'
            ]
            
            if any(indicator in file_stem for indicator in preprocessed_indicators):
                logger.info(f"File {file_path} name indicates preprocessing")
                return True
            
            # Check parent directory for preprocessing indicators
            parent_dir = file_path.parent.name.lower()
            if any(indicator in parent_dir for indicator in ['derivatives', 'processed', 'prep']):
                logger.info(f"File {file_path} in derivatives directory")
                return True
                
            # Check for associated transformation files that would indicate registration
            transform_extensions = ['.txt', '.mat', '.h5', '.xfm']
            for ext in transform_extensions:
                if file_path.with_suffix(ext).exists():
                    logger.info(f"Found transform file for {file_path}")
                    return True
                    
            # If we have SimpleITK, we can try to check some metadata
            if self.libs.get("sitk") is not None:
                try:
                    img = self.libs["sitk"].ReadImage(str(file_path))
                    
                    # Check for specific metadata that might indicate preprocessing
                    for key in img.GetMetaDataKeys():
                        value = img.GetMetaData(key).lower()
                        if any(term in value for term in ['preproc', 'normalized', 'registered']):
                            logger.info(f"File {file_path} has preprocessing metadata")
                            return True
                except:
                    pass
                
            return False
                
        except Exception as e:
            logger.warning(f"Error checking derivative status: {str(e)}")
            return False
    
    def preprocess(self, 
                 scan_file: Optional[Union[str, Path]] = None,
                 bias_correction: bool = True,
                 skull_stripping: bool = True,
                 segmentation: bool = True,
                 normalization: bool = True,
                 check_if_preprocessed: bool = True,
                 **kwargs) -> Dict[str, Any]:
        """Apply standard structural MRI preprocessing pipeline.
        
        Steps include:
        1. Loading scan
        2. Bias field correction
        3. Skull stripping
        4. Tissue segmentation
        5. Spatial normalization
        
        Args:
            scan_file: Path to the scan file (if None, uses first T1w scan)
            bias_correction: Whether to apply bias field correction
            skull_stripping: Whether to apply skull stripping
            segmentation: Whether to perform tissue segmentation
            normalization: Whether to apply spatial normalization
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
                    scan = self._load_scan(scan_file)
                    return {
                        "scan": scan,
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
                # Try to find a T1w scan
                t1_scans = self.dataset.get_structural_scans("T1w")
                if not t1_scans:
                    raise ValueError("No T1w scans found in the dataset")
                scan_file = t1_scans[0]
                logger.info(f"Using first T1w scan: {scan_file}")
            
            # Load the scan
            img, metadata = self._load_scan(scan_file)
            result["original_img"] = img
            result["metadata"] = metadata
            
            # Apply bias field correction
            if bias_correction:
                img = self._apply_bias_correction(img)
                result["bias_corrected_img"] = img
            
            # Apply skull stripping
            if skull_stripping:
                img, mask = self._apply_skull_stripping(img)
                result["brain_img"] = img
                result["brain_mask"] = mask
            
            # Apply tissue segmentation
            if segmentation:
                segments = self._apply_segmentation(img)
                result["segments"] = segments
            
            # Apply spatial normalization
            if normalization:
                norm_img, transform = self._apply_normalization(img)
                result["normalized_img"] = norm_img
                result["normalization_transform"] = transform
            
            # Add metadata
            result["is_preprocessed"] = False
            result["preprocessing_completed"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _load_scan(self, file_path: Union[str, Path]) -> Tuple[Any, Dict[str, Any]]:
        """Load MRI scan and its metadata.
        
        Args:
            file_path: Path to the scan file
            
        Returns:
            Tuple[Any, Dict[str, Any]]: Tuple of (image object, metadata dictionary)
        """
        file_path = Path(file_path)
        logger.info(f"Loading MRI scan from {file_path}")
        
        # Log the preprocessing step
        self.log_preprocessing_step("load_scan", {"file_path": str(file_path)})
        
        try:
            # Use the dataset's method to load the scan
            img, metadata = self.dataset.load_scan(file_path)
            
            if img is None:
                raise ValueError(f"Failed to load scan: {file_path}")
                
            logger.info(f"Successfully loaded {file_path.name}")
            return img, metadata or {}
            
        except Exception as e:
            logger.error(f"Failed to load scan: {str(e)}")
            raise
    
    def _apply_bias_correction(self, img: Any) -> Any:
        """Apply bias field correction to the MRI scan.
        
        Args:
            img: NiBabel or ANTs image object
            
        Returns:
            Any: Bias-corrected image object
        """
        # Log the preprocessing step
        self.log_preprocessing_step("bias_correction", {})
        
        try:
            # If ANTsPy is available, use N4 bias field correction
            if self.libs.get("ants") is not None:
                ants = self.libs["ants"]
                logger.info("Applying N4 bias field correction using ANTsPy")
                
                # Convert to ANTs image if needed
                if not isinstance(img, ants.core.ants_image.ANTsImage):
                    img_data = img.get_fdata()
                    img_affine = img.affine
                    ants_img = ants.from_numpy(img_data, img_affine)
                else:
                    ants_img = img
                
                # Apply N4 bias field correction
                bias_corrected = ants.n4_bias_field_correction(ants_img)
                logger.info("Bias field correction completed")
                
                return bias_corrected
                
            # If ANTsPy is not available but Nilearn is, try SimpleITK through Nilearn
            elif self.libs.get("nilearn") is not None:
                try:
                    import SimpleITK as sitk
                    from nilearn.image import new_img_like
                    
                    logger.info("Applying N4 bias field correction using SimpleITK")
                    
                    # Convert to SimpleITK image
                    img_data = img.get_fdata()
                    sitk_img = sitk.GetImageFromArray(img_data.astype(np.float32))
                    
                    # Apply N4 bias field correction
                    corrector = sitk.N4BiasFieldCorrectionImageFilter()
                    sitk_corrected = corrector.Execute(sitk_img)
                    
                    # Convert back to NiBabel image
                    corrected_data = sitk.GetArrayFromImage(sitk_corrected)
                    bias_corrected = new_img_like(img, corrected_data)
                    
                    logger.info("Bias field correction completed")
                    return bias_corrected
                    
                except ImportError:
                    logger.warning("SimpleITK not available. Skipping bias correction.")
                    return img
            
            # If neither is available, return the original image
            logger.warning("No bias correction libraries available. Returning original image.")
            return img
            
        except Exception as e:
            logger.error(f"Failed to apply bias correction: {str(e)}")
            return img
    
    def _apply_skull_stripping(self, img: Any) -> Tuple[Any, Any]:
        """Apply skull stripping to the MRI scan.
        
        Args:
            img: NiBabel or ANTs image object
            
        Returns:
            Tuple[Any, Any]: Tuple of (brain image, brain mask)
        """
        # Log the preprocessing step
        self.log_preprocessing_step("skull_stripping", {})
        
        try:
            # If ANTsPy is available, use ANTs brain extraction
            if self.libs.get("ants") is not None:
                ants = self.libs["ants"]
                logger.info("Applying skull stripping using ANTsPy")
                
                # Convert to ANTs image if needed
                if not isinstance(img, ants.core.ants_image.ANTsImage):
                    img_data = img.get_fdata()
                    img_affine = img.affine
                    ants_img = ants.from_numpy(img_data, img_affine)
                else:
                    ants_img = img
                
                # Apply brain extraction
                brain_extract = ants.get_ants_data("kirby")
                brain_template = ants.image_read(brain_extract["t1w"])
                brain_probability_mask = ants.image_read(brain_extract["probability_mask"])
                
                # Run brain extraction
                brain_extraction = ants.brain_extraction(
                    image=ants_img,
                    brain_template=brain_template,
                    brain_probability_mask=brain_probability_mask,
                    extraction_label=1
                )
                
                brain_img = brain_extraction["BrainExtractionBrain"]
                brain_mask = brain_extraction["BrainExtractionMask"]
                
                logger.info("Skull stripping completed")
                return brain_img, brain_mask
                
            # If ANTsPy is not available but Nilearn is, try BET through Nilearn
            elif self.libs.get("nilearn") is not None:
                try:
                    from nilearn.masking import compute_brain_mask
                    
                    logger.info("Computing brain mask using Nilearn")
                    
                    # Compute brain mask
                    brain_mask = compute_brain_mask(img)
                    
                    # Apply mask to image
                    from nilearn.masking import apply_mask
                    brain_data = apply_mask(img, brain_mask)
                    
                    # Create brain image
                    nib = self.libs["nibabel"]
                    brain_img = nib.Nifti1Image(brain_data, img.affine)
                    
                    logger.info("Skull stripping completed")
                    return brain_img, brain_mask
                    
                except Exception as e:
                    logger.warning(f"Failed to use Nilearn for skull stripping: {str(e)}")
                    return img, None
            
            # If neither is available, return the original image
            logger.warning("No skull stripping libraries available. Returning original image.")
            return img, None
            
        except Exception as e:
            logger.error(f"Failed to apply skull stripping: {str(e)}")
            return img, None
    
    def _apply_segmentation(self, img: Any) -> Dict[str, Any]:
        """Segment the MRI scan into different tissue types.
        
        Args:
            img: NiBabel or ANTs image object
            
        Returns:
            Dict[str, Any]: Dictionary of segmented tissues
        """
        # Log the preprocessing step
        self.log_preprocessing_step("segmentation", {})
        
        try:
            # If ANTsPy is available, use ANTs segmentation
            if self.libs.get("ants") is not None:
                ants = self.libs["ants"]
                logger.info("Applying tissue segmentation using ANTsPy")
                
                # Convert to ANTs image if needed
                if not isinstance(img, ants.core.ants_image.ANTsImage):
                    img_data = img.get_fdata()
                    img_affine = img.affine
                    ants_img = ants.from_numpy(img_data, img_affine)
                else:
                    ants_img = img
                
                # Apply Atropos segmentation
                segmentation = ants.atropos(ants_img, 
                                           m='[0.2,1x1x1]', 
                                           c='[2,0]',
                                           i='kmeans[3]')
                
                # Extract segmentation results
                segments = {
                    "segmentation_image": segmentation["segmentation"],
                    "csf": segmentation["probabilityimages"][0],
                    "gm": segmentation["probabilityimages"][1],
                    "wm": segmentation["probabilityimages"][2]
                }
                
                logger.info("Segmentation completed")
                return segments
                
            # If ANTsPy is not available but Nilearn is, try FAST through Nilearn
            elif self.libs.get("nilearn") is not None:
                try:
                    from nilearn.regions import connected_regions
                    
                    logger.info("Computing simplified segmentation using Nilearn")
                    
                    # Compute brain mask
                    from nilearn.masking import compute_brain_mask
                    brain_mask = compute_brain_mask(img)
                    
                    # Use clustering for a simple segmentation
                    from nilearn.regions import RegionExtractor
                    from nilearn._utils import check_niimg
                    from sklearn.feature_extraction import image
                    
                    # Extract connected components from the image
                    img_checked = check_niimg(img)
                    img_data = img_checked.get_fdata()
                    
                    # Normalize data
                    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
                    
                    # Apply mask
                    mask_data = brain_mask.get_fdata().astype(bool)
                    img_data[~mask_data] = 0
                    
                    # Simple kmeans clustering for 3 tissue types
                    from sklearn.cluster import KMeans
                    img_data_flat = img_data[mask_data].reshape(-1, 1)
                    kmeans = KMeans(n_clusters=3, random_state=42)
                    labels = kmeans.fit_predict(img_data_flat)
                    
                    # Reconstruct segmentation image
                    seg_data = np.zeros_like(img_data)
                    seg_data[mask_data] = labels + 1  # 1=CSF, 2=GM, 3=WM
                    
                    # Create segmentation image
                    nib = self.libs["nibabel"]
                    seg_img = nib.Nifti1Image(seg_data, img.affine)
                    
                    # Create probability maps (simple binary masks for simplicity)
                    csf_data = np.zeros_like(img_data)
                    csf_data[mask_data] = (labels == 0).astype(float)
                    csf_img = nib.Nifti1Image(csf_data, img.affine)
                    
                    gm_data = np.zeros_like(img_data)
                    gm_data[mask_data] = (labels == 1).astype(float)
                    gm_img = nib.Nifti1Image(gm_data, img.affine)
                    
                    wm_data = np.zeros_like(img_data)
                    wm_data[mask_data] = (labels == 2).astype(float)
                    wm_img = nib.Nifti1Image(wm_data, img.affine)
                    
                    segments = {
                        "segmentation_image": seg_img,
                        "csf": csf_img,
                        "gm": gm_img,
                        "wm": wm_img
                    }
                    
                    logger.info("Simple segmentation completed")
                    return segments
                    
                except Exception as e:
                    logger.warning(f"Failed to use Nilearn for segmentation: {str(e)}")
                    return {}
            
            # If neither is available, return empty dictionary
            logger.warning("No segmentation libraries available.")
            return {}
            
        except Exception as e:
            logger.error(f"Failed to apply segmentation: {str(e)}")
            return {}
    
    def _apply_normalization(self, img: Any) -> Tuple[Any, Any]:
        """Apply spatial normalization to the MRI scan.
        
        Args:
            img: NiBabel or ANTs image object
            
        Returns:
            Tuple[Any, Any]: Tuple of (normalized image, transformation)
        """
        # Log the preprocessing step
        self.log_preprocessing_step("normalization", {})
        
        try:
            # If ANTsPy is available, use ANTs registration
            if self.libs.get("ants") is not None:
                ants = self.libs["ants"]
                logger.info("Applying spatial normalization using ANTsPy")
                
                # Convert to ANTs image if needed
                if not isinstance(img, ants.core.ants_image.ANTsImage):
                    img_data = img.get_fdata()
                    img_affine = img.affine
                    ants_img = ants.from_numpy(img_data, img_affine)
                else:
                    ants_img = img
                
                # Get MNI template
                mni_template = ants.get_ants_data("mni")["mni"]
                mni_img = ants.image_read(mni_template)
                
                # Apply registration
                registration = ants.registration(
                    fixed=mni_img,
                    moving=ants_img,
                    type_of_transform='SyN'
                )
                
                # Extract registration results
                normalized_img = registration["warpedmovout"]
                transform = [registration["fwdtransforms"], registration["invtransforms"]]
                
                logger.info("Spatial normalization completed")
                return normalized_img, transform
                
            # If ANTsPy is not available but Nilearn is, use Nilearn for registration
            elif self.libs.get("nilearn") is not None:
                try:
                    from nilearn.image import resample_to_img
                    from nilearn.datasets import load_mni152_template
                    
                    logger.info("Applying spatial normalization using Nilearn")
                    
                    # Load MNI template
                    mni_template = load_mni152_template()
                    
                    # Apply registration
                    normalized_img = resample_to_img(img, mni_template)
                    
                    logger.info("Spatial normalization completed")
                    return normalized_img, None
                    
                except Exception as e:
                    logger.warning(f"Failed to use Nilearn for normalization: {str(e)}")
                    return img, None
            
            # If neither is available, return the original image
            logger.warning("No normalization libraries available. Returning original image.")
            return img, None
            
        except Exception as e:
            logger.error(f"Failed to apply normalization: {str(e)}")
            return img, None
    
    # Additional MRI preprocessing methods
    
    def extract_brain_region(self, img: Any, region_name: str) -> Any:
        """Extract a specific brain region using an atlas.
        
        Args:
            img: NiBabel or ANTs image object
            region_name: Name of the region to extract
            
        Returns:
            Any: Image of the extracted region
        """
        self.log_preprocessing_step("extract_brain_region", {"region_name": region_name})
        
        try:
            # Check if Nilearn is available
            if self.libs.get("nilearn") is None:
                logger.error("Nilearn is required for brain region extraction")
                return None
                
            nilearn = self.libs["nilearn"]
            from nilearn import datasets, image, masking
            
            logger.info(f"Extracting brain region: {region_name}")
            
            # Load Harvard-Oxford atlas
            atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            atlas_img = atlas['maps']
            labels = atlas['labels']
            
            # Find the region of interest
            if region_name not in labels:
                logger.error(f"Region {region_name} not found in the atlas")
                return None
                
            region_index = labels.index(region_name)
            
            # Create binary mask for the region
            mask_data = (atlas_img.get_fdata() == region_index)
            mask_img = image.new_img_like(atlas_img, mask_data.astype(int))
            
            # Resample mask to the image space
            mask_resampled = image.resample_to_img(mask_img, img)
            
            # Apply mask to the image
            region_img = masking.apply_mask(img, mask_resampled)
            
            logger.info(f"Successfully extracted region: {region_name}")
            return region_img
            
        except Exception as e:
            logger.error(f"Failed to extract brain region: {str(e)}")
            return None
    
    def calculate_brain_volumes(self, segments: Dict[str, Any]) -> Dict[str, float]:
        """Calculate volumes of different brain tissues.
        
        Args:
            segments: Dictionary of segmented tissues from _apply_segmentation
            
        Returns:
            Dict[str, float]: Dictionary of tissue volumes in mm³
        """
        self.log_preprocessing_step("calculate_brain_volumes", {})
        
        try:
            if not segments:
                logger.error("No segmentation data provided")
                return {}
                
            nib = self.libs["nibabel"]
            
            # Initialize volume dictionary
            volumes = {}
            
            # Calculate volume for each tissue type
            for tissue, img in segments.items():
                if tissue == "segmentation_image":
                    continue
                    
                # Get voxel dimensions
                if hasattr(img, 'header'):
                    voxel_size = img.header.get_zooms()
                else:
                    # For ANTs images
                    voxel_size = img.spacing
                
                # Calculate voxel volume in mm³
                voxel_volume = np.prod(voxel_size)
                
                # Count non-zero voxels
                if isinstance(img, nib.nifti1.Nifti1Image):
                    data = img.get_fdata()
                else:
                    # For ANTs images
                    data = img.numpy()
                
                # Calculate volume
                tissue_volume = np.sum(data > 0.5) * voxel_volume
                volumes[tissue] = tissue_volume
            
            # Add total brain volume
            if "gm" in volumes and "wm" in volumes:
                volumes["total_brain"] = volumes["gm"] + volumes["wm"]
                
            logger.info(f"Calculated brain volumes: {volumes}")
            return volumes
            
        except Exception as e:
            logger.error(f"Failed to calculate brain volumes: {str(e)}")
            return {} 