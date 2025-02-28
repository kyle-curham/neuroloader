"""EEG data preprocessing module"""

import os
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import pandas as pd

from ..loaders.eeg_loader import EEGDataset
from .base_processor import BaseProcessor
from ..logger import get_logger

# Get logger for this module
logger = get_logger("processors.eeg")

class EEGProcessor(BaseProcessor):
    """Processor for EEG data preprocessing.
    
    This class implements standard preprocessing steps for EEG data
    following MNE-Python best practices.
    """
    
    def __init__(self, dataset: Optional[Any] = None):
        """Initialize the EEG processor.
        
        Args:
            dataset: The dataset to process (optional)
        """
        super().__init__(dataset)
        self.dataset = dataset
        
        # Try to import MNE
        try:
            import mne
            self.mne = mne
            logger.info("Successfully imported MNE")
        except ImportError:
            logger.warning("MNE is not installed. EEG processing functionality will be limited.")
            self.mne = None
    
    def check_derivative_status(self, file_path: Union[str, Path]) -> bool:
        """Check if the EEG data appears to be preprocessed.
        
        This method checks EEG-specific indicators of preprocessing.
        
        Args:
            file_path: Path to the EEG recording file
            
        Returns:
            bool: True if the file appears to be preprocessed, False otherwise
        """
        # Convert to Path object
        file_path = Path(file_path)
        
        if not self.mne:
            logger.warning("MNE not available for derivative status check")
            return False
            
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist")
            return False
            
        try:
            # Check if this is a prepared epochs file
            if file_path.suffix == '.fif':
                try:
                    # Try to load as epochs
                    self.mne.read_epochs(file_path, preload=False)
                    logger.info(f"File {file_path} is a preprocessed epochs file")
                    return True
                except:
                    # Check if it contains processing history
                    raw = self.mne.io.read_raw_fif(file_path, preload=False)
                    if hasattr(raw, 'annotations') and len(raw.annotations) > 0:
                        # Check annotations for preprocessing markers
                        for note in raw.annotations.description:
                            if any(marker in note.lower() for marker in 
                                   ['filter', 'ica', 'preprocess', 'clean', 'reference']):
                                logger.info(f"File {file_path} has preprocessing annotations")
                                return True
            
            # For EEGLAB .set files
            if file_path.suffix == '.set':
                # Load without preloading data
                raw = self.mne.io.read_raw_eeglab(file_path, preload=False)
                
                # Check if ICA components exist (sign of preprocessing)
                if hasattr(raw, 'ica') or hasattr(raw, '_ica'):
                    logger.info(f"File {file_path} has ICA components")
                    return True
                    
            # Check file naming conventions that indicate preprocessing
            file_stem = file_path.stem.lower()
            preprocessed_indicators = ['proc', 'clean', 'filt', 'ica', 'preproc', 'epoched']
            if any(indicator in file_stem for indicator in preprocessed_indicators):
                logger.info(f"File {file_path} name indicates preprocessing")
                return True
                
            return False
                
        except Exception as e:
            logger.warning(f"Error checking derivative status: {str(e)}")
            return False

    def preprocess(self, 
                  recording_file: Union[str, Path],
                  filter_params: Optional[Dict[str, Any]] = None,
                  reference_type: str = 'average',
                  ica_params: Optional[Dict[str, Any]] = None,
                  epoch_params: Optional[Dict[str, Any]] = None,
                  check_if_preprocessed: bool = True) -> Dict[str, Any]:
        """Preprocess EEG data.
        
        Args:
            recording_file: Path to the EEG recording file
            filter_params: Dictionary with filter settings (l_freq, h_freq, notch_freqs)
            reference_type: Reference type ('average', 'mastoids', etc.)
            ica_params: Dictionary with ICA settings (n_components, method, etc.)
            epoch_params: Dictionary with epoching parameters (tmin, tmax, event_id)
            check_if_preprocessed: Whether to check if file appears already preprocessed
            
        Returns:
            Dict[str, Any]: Dictionary with preprocessing results
        """
        self.log_preprocessing_step("preprocess", locals())
        
        # Check if MNE is available
        if self.mne is None:
            logger.error("MNE is not available. Cannot preprocess EEG data.")
            return {"error": "MNE not available"}
        
        # Check if the file is already preprocessed
        if check_if_preprocessed:
            is_preprocessed = self.check_derivative_status(recording_file)
            if is_preprocessed:
                logger.info(f"File {recording_file} appears to be preprocessed already. Skipping preprocessing.")
                
                # Try to load the data without additional processing
                try:
                    # Attempt to load as epochs first
                    try:
                        epochs = self.mne.read_epochs(recording_file)
                        return {
                            "epochs": epochs,
                            "is_preprocessed": True,
                            "preprocessing_skipped": True
                        }
                    except:
                        # If not epochs, load as raw
                        raw = self._load_raw_data(recording_file)
                        return {
                            "raw": raw,
                            "is_preprocessed": True,
                            "preprocessing_skipped": True
                        }
                except Exception as e:
                    logger.error(f"Error loading preprocessed file: {str(e)}")
                    # Continue with preprocessing as fallback
        
        # Default filter settings if not provided
        if filter_params is None:
            filter_params = {
                'l_freq': 1.0,       # High-pass filter at 1 Hz
                'h_freq': 40.0,      # Low-pass filter at 40 Hz
                'notch_freqs': [50.0, 60.0]  # Notch filters for line noise
            }
            
        # Default ICA settings if not provided
        if ica_params is None:
            ica_params = {
                'n_components': 0.95,   # Retain components explaining 95% variance
                'method': 'fastica',
                'random_state': 42
            }
            
        # Default epoch settings if not provided
        if epoch_params is None:
            epoch_params = {
                'tmin': -0.2,         # 200ms before event
                'tmax': 0.8,          # 800ms after event
                'baseline': (None, 0), # Baseline correction from start to event
                'event_id': None      # Use all event types
            }
            
        try:
            # 1. Load raw data
            raw = self._load_raw_data(recording_file)
            logger.info(f"Loaded raw data: {len(raw.times)/raw.info['sfreq']:.1f}s at {raw.info['sfreq']}Hz")
            
            # 2. Apply filtering
            raw_filtered = self._apply_filtering(raw, 
                                              l_freq=filter_params.get('l_freq'), 
                                              h_freq=filter_params.get('h_freq'),
                                              notch_freqs=filter_params.get('notch_freqs'))
            
            # 3. Apply reference
            raw_referenced = self._apply_reference(raw_filtered, reference_type)
            
            # 4. Detect and interpolate bad channels
            raw_clean = self._detect_bad_channels(raw_referenced)
            
            # 5. Apply ICA for artifact removal
            raw_ica = self._apply_ica(raw_clean, **ica_params)
            
            # 6. Get events
            events = self._get_events(raw_ica, recording_file, epoch_params.get('event_id'))
            
            # 7. Create epochs
            epochs = self._create_epochs(raw_ica, events,
                                      tmin=epoch_params.get('tmin'),
                                      tmax=epoch_params.get('tmax'),
                                      baseline=epoch_params.get('baseline'),
                                      event_id=epoch_params.get('event_id'))
            
            # Create evoked objects (averaged epochs)
            evoked = {}
            if isinstance(epoch_params.get('event_id'), dict):
                for event_name in epoch_params.get('event_id', {}).keys():
                    try:
                        evoked[event_name] = epochs[event_name].average()
                    except:
                        logger.warning(f"Could not create evoked response for {event_name}")
            else:
                # Create a single evoked object if no event_id was specified
                evoked['average'] = epochs.average()
                
            # Return results
            results = {
                "raw": raw,
                "raw_filtered": raw_filtered,
                "raw_referenced": raw_referenced,
                "raw_clean": raw_clean,
                "raw_ica": raw_ica,
                "events": events,
                "epochs": epochs,
                "evoked": evoked,
                "is_preprocessed": False,
                "preprocessing_completed": True
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _load_raw_data(self, file_path: Union[str, Path]) -> Any:
        """Load raw EEG data using MNE.
        
        Args:
            file_path: Path to the EEG file
            
        Returns:
            mne.io.Raw: Raw MNE object
        """
        file_path = Path(file_path)
        logger.info(f"Loading EEG data from {file_path}")
        
        # Log the preprocessing step
        self.log_preprocessing_step("load_raw_data", {"file_path": str(file_path)})
        
        try:
            # Determine file format and use appropriate reader
            if file_path.suffix == '.fif':
                raw = self.mne.io.read_raw_fif(file_path, preload=True)
            elif file_path.suffix == '.edf':
                raw = self.mne.io.read_raw_edf(file_path, preload=True)
            elif file_path.suffix == '.bdf':
                raw = self.mne.io.read_raw_bdf(file_path, preload=True)
            elif file_path.suffix == '.set':
                raw = self.mne.io.read_raw_eeglab(file_path, preload=True)
            elif file_path.suffix in ('.cnt', '.avg', '.eeg'):
                raw = self.mne.io.read_raw_cnt(file_path, preload=True)
            elif file_path.suffix == '.vhdr':
                raw = self.mne.io.read_raw_brainvision(file_path, preload=True)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
            logger.info(f"Successfully loaded {file_path.name} with {len(raw.ch_names)} channels")
            return raw
            
        except Exception as e:
            logger.error(f"Failed to load EEG data: {str(e)}")
            raise
    
    def _apply_filtering(self, raw: Any, l_freq: float, h_freq: float, notch_freqs: List[float]) -> Any:
        """Apply frequency filters to the raw data.
        
        Args:
            raw: MNE Raw object
            l_freq: Low cutoff frequency (highpass)
            h_freq: High cutoff frequency (lowpass)
            notch_freqs: Frequencies to notch filter
            
        Returns:
            mne.io.Raw: Filtered Raw object
        """
        # Log the preprocessing step
        self.log_preprocessing_step("apply_filtering", {"l_freq": l_freq, "h_freq": h_freq, "notch_freqs": notch_freqs})
        
        # Apply bandpass or highpass/lowpass filter
        if l_freq is not None or h_freq is not None:
            logger.info(f"Applying filter with l_freq={l_freq}, h_freq={h_freq}")
            raw.filter(l_freq=l_freq, h_freq=h_freq)
        
        # Apply notch filter if specified
        if notch_freqs is not None:
            logger.info(f"Applying notch filters at {notch_freqs} Hz")
            for freq in notch_freqs:
                raw.notch_filter(freqs=freq)
        
        return raw
    
    def _apply_reference(self, raw: Any, reference: str) -> Any:
        """Apply re-referencing to the raw data.
        
        Args:
            raw: MNE Raw object
            reference: Reference type ('average', 'mastoids', etc.)
                
        Returns:
            mne.io.Raw: Re-referenced Raw object
        """
        # Log the preprocessing step
        self.log_preprocessing_step("apply_reference", {"reference": reference})
        
        try:
            if reference == "average":
                logger.info("Applying average reference")
                raw.set_eeg_reference('average', projection=False)
            elif reference == "mastoids":
                logger.info("Applying mastoid reference")
                raw.set_eeg_reference(['M1', 'M2'], projection=False)
            else:
                logger.info(f"Applying {reference} reference")
                raw.set_eeg_reference(reference, projection=False)
                
            return raw
            
        except Exception as e:
            logger.error(f"Failed to apply reference: {str(e)}")
            # Continue with original reference
            return raw
    
    def _detect_bad_channels(self, raw: Any) -> Any:
        """Detect and interpolate bad channels.
        
        Args:
            raw: MNE Raw object
                
        Returns:
            mne.io.Raw: Processed Raw object
        """
        # Log the preprocessing step
        self.log_preprocessing_step("detect_bad_channels", {})
        
        try:
            # Use MNE's function to find bad channels
            bads = self.mne.preprocessing.find_bad_channels_maxwell(raw, 
                                                               ignore_ref=True, 
                                                               return_scores=False)
            
            # Mark the channels as bad
            raw.info['bads'] = bads
            logger.info(f"Detected {len(bads)} bad channels: {bads}")
            
            # Interpolate bad channels if there are any
            if bads:
                raw = raw.interpolate_bads()
                logger.info("Interpolated bad channels")
                
            return raw
            
        except Exception as e:
            logger.error(f"Failed to detect bad channels: {str(e)}")
            return raw
    
    def _apply_ica(self, raw: Any, n_components: float, method: str, random_state: int) -> Any:
        """Apply Independent Component Analysis for artifact removal.
        
        Args:
            raw: MNE Raw object
            n_components: Number of components to keep
            method: ICA method ('fastica', 'picard', etc.)
            random_state: Random state for reproducibility
            
        Returns:
            mne.io.Raw: Processed Raw object
        """
        # Log the preprocessing step
        self.log_preprocessing_step("apply_ica", {"n_components": n_components, "method": method, "random_state": random_state})
        
        try:
            # Create ICA object
            ica = self.mne.preprocessing.ICA(
                n_components=n_components,
                random_state=random_state,
                method=method
            )
            
            # Fit ICA
            logger.info(f"Fitting ICA with {n_components} components using {method}")
            ica.fit(raw)
            
            # Find EOG artifacts
            logger.info("Detecting eye-related components")
            eog_indices, eog_scores = ica.find_bads_eog(raw)
            logger.info(f"Detected {len(eog_indices)} EOG components: {eog_indices}")
            
            # Find ECG artifacts if available
            ecg_indices = []
            if any(ch_name.startswith('ECG') for ch_name in raw.ch_names):
                logger.info("Detecting heart-related components")
                ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
                logger.info(f"Detected {len(ecg_indices)} ECG components: {ecg_indices}")
            
            # Mark components as bad
            ica.exclude = eog_indices + ecg_indices
            
            # Apply ICA to remove artifacts
            logger.info(f"Applying ICA to remove {len(ica.exclude)} artifacts")
            raw_ica = raw.copy()
            ica.apply(raw_ica)
            
            return raw_ica
            
        except Exception as e:
            logger.error(f"Failed to apply ICA: {str(e)}")
            return raw
    
    def _get_events(self, raw: Any, recording_file: Union[str, Path], event_id: Optional[Union[dict, list]] = None) -> np.ndarray:
        """Get events from the raw data or events file.
        
        Args:
            raw: MNE Raw object
            recording_file: Path to the recording file
            event_id: Event IDs to use
                
        Returns:
            np.ndarray: Events array
        """
        # Try to get events from the dataset
        events_df = self.dataset.get_events_dataframe(recording_file)
        
        # Log the preprocessing step
        self.log_preprocessing_step("get_events", {"recording_file": str(recording_file)})
        
        # If events are available in the dataset
        if not events_df.empty:
            logger.info(f"Using {len(events_df)} events from dataset")
            
            # Convert DataFrame to MNE events format (sample, 0, event_id)
            # Assuming events_df has 'onset' and 'event_type' columns
            if 'onset' in events_df.columns and 'event_type' in events_df.columns:
                # Convert onsets to samples
                sample_rate = raw.info['sfreq']
                sample_indices = (events_df['onset'] * sample_rate).astype(int)
                
                # Create mapping of event types to integers
                event_types = events_df['event_type'].unique()
                event_id = {str(etype): i+1 for i, etype in enumerate(event_types)}
                
                # Create events array
                events = np.zeros((len(events_df), 3), dtype=int)
                events[:, 0] = sample_indices
                events[:, 2] = [event_id[str(etype)] for etype in events_df['event_type']]
                
                logger.info(f"Created events array with {len(events)} events")
                return events
        
        # If no events from dataset, try to find them in the raw data
        try:
            logger.info("Finding events from raw data")
            events = self.mne.find_events(raw, stim_channel='STI', verbose=False)
            logger.info(f"Found {len(events)} events in raw data")
            return events
        except Exception as e:
            logger.error(f"Failed to find events: {str(e)}")
            # Create dummy event at the beginning
            logger.warning("Creating dummy event at the beginning")
            return np.array([[0, 0, 1]])
    
    def _create_epochs(self, raw: Any, events: np.ndarray, tmin: float, tmax: float, baseline: Tuple[Optional[float], float], event_id: Optional[Union[dict, list]] = None) -> Any:
        """Create epochs from raw data.
        
        Args:
            raw: MNE Raw object
            events: Events array
            tmin: Start time of the epoch
            tmax: End time of the epoch
            baseline: Baseline correction
            event_id: Event IDs to use
            
        Returns:
            mne.Epochs: Epochs object
        """
        # Log the preprocessing step
        self.log_preprocessing_step("create_epochs", {"tmin": tmin, "tmax": tmax, "baseline": baseline, "event_id": event_id})
        
        try:
            # Create epochs
            logger.info(f"Creating epochs with tmin={tmin}, tmax={tmax}")
            epochs = self.mne.Epochs(
                raw, 
                events, 
                event_id=event_id,
                tmin=tmin, 
                tmax=tmax,
                baseline=baseline,
                preload=True
            )
            
            logger.info(f"Created {len(epochs)} epochs")
            return epochs
            
        except Exception as e:
            logger.error(f"Failed to create epochs: {str(e)}")
            return None
    
    # Additional EEG preprocessing methods
    
    def compute_power_spectrum(self, raw_or_epochs: Any, fmin: float = 0, fmax: float = 50, 
                              n_fft: int = 2048) -> Dict[str, Any]:
        """Compute power spectral density.
        
        Args:
            raw_or_epochs: MNE Raw or Epochs object
            fmin: Minimum frequency
            fmax: Maximum frequency
            n_fft: Number of points for FFT
            
        Returns:
            Dict[str, Any]: PSD data and frequencies
        """
        self.log_preprocessing_step("compute_power_spectrum", 
                               {"fmin": fmin, "fmax": fmax, "n_fft": n_fft})
        
        try:
            logger.info(f"Computing power spectrum from {fmin}-{fmax} Hz")
            psd, freqs = raw_or_epochs.compute_psd(fmin=fmin, fmax=fmax, n_fft=n_fft).get_data(return_freqs=True)
            
            return {
                "psd": psd,
                "freqs": freqs
            }
            
        except Exception as e:
            logger.error(f"Failed to compute power spectrum: {str(e)}")
            return {}
    
    def compute_connectivity(self, epochs: Any, method: str = 'wpli', 
                            fmin: float = 8, fmax: float = 13) -> Dict[str, Any]:
        """Compute connectivity between channels.
        
        Args:
            epochs: MNE Epochs object
            method: Connectivity method ('coh', 'plv', 'wpli', etc.)
            fmin: Minimum frequency
            fmax: Maximum frequency
            
        Returns:
            Dict[str, Any]: Connectivity data
        """
        self.log_preprocessing_step("compute_connectivity", 
                               {"method": method, "fmin": fmin, "fmax": fmax})
        
        try:
            # Import from mne_connectivity instead of mne.connectivity
            try:
                import mne_connectivity
                from mne_connectivity import spectral_connectivity
                logger.info("Successfully imported mne_connectivity")
            except ImportError:
                logger.error("mne_connectivity is not installed. Please install it with: pip install mne_connectivity")
                return {}
            
            logger.info(f"Computing {method} connectivity between {fmin}-{fmax} Hz")
            con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                epochs, 
                method=method,
                mode='multitaper',
                sfreq=epochs.info['sfreq'],
                fmin=fmin, 
                fmax=fmax,
                faverage=True
            )
            
            return {
                "connectivity": con,
                "freqs": freqs,
                "times": times
            }
            
        except Exception as e:
            logger.error(f"Failed to compute connectivity: {str(e)}")
            return {} 