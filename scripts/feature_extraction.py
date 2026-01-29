#!/usr/bin/env python3
# coding: utf-8

"""
===============================================================================
ROBUST FEATURE EXTRACTION SCRIPT FOR IRBD DETECTION
===============================================================================

This script extracts SSL-Wearables embeddings from preprocessed accelerometer data.
Supports multiple preprocessing versions (v0, v1, v1t, vvt) via command-line argument.

KEY FEATURES:
- Uses HAR-Net10 model pre-trained on UK Biobank data (100k participants)
- Extracts 1024-dimensional embeddings from 10-second windows
- Robust error handling: skips bad nights instead of failing entire participant
- Memory-efficient batch processing with GPU support
- Comprehensive logging and statistics tracking

PREPROCESSING COMPATIBILITY:
- Expects HDF5 files with structure: night_0/accel, night_0/timestamps, etc.
- Assumes 30Hz sampling rate (300 samples = 10 seconds)
- Handles variable-length nights with padding and masking

SSL-WEARABLES REFERENCE:
Yuan, H., Chan, S., Creagh, A. P., et al. (2024). Self-supervised learning 
for human activity recognition using 700,000 person-days of wearable data. 
NPJ Digital Medicine, 7(1), 91.
===============================================================================
"""

import argparse
import os
import sys
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import glob
import traceback
import json
import gc  # For explicit memory management

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Configure matplotlib and seaborn for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# PYTORCH AND SSL-WEARABLES MODEL LOADING
# ============================================================================

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    print(f"PyTorch version: {torch.__version__}")

    # Check for CUDA availability (GPU acceleration)
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available - will use CPU (slower)")

except ImportError as e:
    print(f"Error importing PyTorch: {e}")
    print("Please install PyTorch with: pip install torch")
    sys.exit(1)

# Try to load SSL-Wearables HAR-Net10 model
# This model expects 10-second windows (300 samples at 30Hz)
# Input shape: (batch_size, 3, 300) where 3 = XYZ channels
# Output: 1024-dimensional feature embeddings
try:
    print("Loading SSL-Wearables HAR-Net10 model...")
    print("  - Model: ResNet-based architecture")
    print("  - Pre-training: UK Biobank (100k participants)")
    print("  - Window size: 10 seconds (300 samples at 30Hz)")
    print("  - Feature dimension: 1024")
    
    # Load model from torch.hub
    # trust_repo=True is required for loading custom models
    ssl_model = torch.hub.load('OxWearables/ssl-wearables', 'harnet10', 
                                pretrained=True, trust_repo=True)
    print("SSL-Wearables model loaded successfully")
    
except Exception as e:
    print(f"Error loading SSL-Wearables model: {e}")
    print("Please check internet connection and torch.hub access")
    print("If behind a firewall, you may need to set HTTP_PROXY and HTTPS_PROXY")
    sys.exit(1)


# ============================================================================
# ROBUSTNESS HELPER FUNCTIONS
# ============================================================================

def validate_sampling_rate(timestamps, expected_hz=30.0, tolerance=1.0):
    """
    Validate that the sampling rate is close to expected value.
    
    Args:
        timestamps (list/array): List of timestamp strings
        expected_hz (float): Expected sampling rate in Hz
        tolerance (float): Acceptable deviation in Hz
        
    Returns:
        tuple: (is_valid, actual_hz, message)
    """
    if len(timestamps) < 2:
        return True, None, "Insufficient timestamps for rate validation"
    
    try:
        # Sample first 100 intervals to estimate sampling rate
        sample_size = min(100, len(timestamps) - 1)
        intervals = []
        
        for i in range(sample_size):
            t1 = pd.Timestamp(timestamps[i])
            t2 = pd.Timestamp(timestamps[i + 1])
            intervals.append((t2 - t1).total_seconds())
        
        avg_interval = np.mean(intervals)
        actual_hz = 1 / avg_interval if avg_interval > 0 else 0
        
        # Check if within tolerance
        is_valid = abs(actual_hz - expected_hz) <= tolerance
        
        if is_valid:
            message = f"Sampling rate OK: {actual_hz:.1f}Hz (expected {expected_hz}Hz)"
        else:
            message = f"Sampling rate WARNING: {actual_hz:.1f}Hz (expected {expected_hz}±{tolerance}Hz)"
        
        return is_valid, actual_hz, message
        
    except Exception as e:
        return False, None, f"Error validating sampling rate: {e}"


def validate_data_consistency(accel_data, timestamps):
    """
    Validate consistency between accelerometer data and timestamps.
    
    Args:
        accel_data (np.ndarray): Accelerometer data (N, 3)
        timestamps (list/array): Timestamp strings (N,)
        
    Returns:
        tuple: (is_valid, message)
    """
    issues = []
    
    # Check shape
    if accel_data.ndim != 2:
        issues.append(f"Invalid accel dimensions: {accel_data.ndim}D (expected 2D)")
    elif accel_data.shape[1] != 3:
        issues.append(f"Invalid accel channels: {accel_data.shape[1]} (expected 3 for XYZ)")
    
    # Check length match
    if len(accel_data) != len(timestamps):
        issues.append(
            f"Length mismatch: {len(accel_data)} accel samples vs "
            f"{len(timestamps)} timestamps"
        )
    
    # Check for NaN or Inf
    if np.any(np.isnan(accel_data)):
        nan_count = np.sum(np.isnan(accel_data))
        issues.append(f"Contains {nan_count} NaN values")
    
    if np.any(np.isinf(accel_data)):
        inf_count = np.sum(np.isinf(accel_data))
        issues.append(f"Contains {inf_count} Inf values")
    
    # Check for reasonable acceleration range (typically -20 to +20 g)
    if accel_data.size > 0:
        min_val = np.min(accel_data)
        max_val = np.max(accel_data)
        if min_val < -50 or max_val > 50:
            issues.append(
                f"Suspicious acceleration range: [{min_val:.1f}, {max_val:.1f}] "
                f"(expected roughly -20 to +20 g)"
            )
    
    if issues:
        return False, "; ".join(issues)
    else:
        return True, "Data consistency OK"


# ============================================================================
# FEATURE EXTRACTION PIPELINE CLASS
# ============================================================================

class RobustFeatureExtractionPipeline:
    """
    Robust feature extraction pipeline with defensive checks and validation.
    
    ROBUSTNESS FEATURES:
    --------------------
    1. Version verification: Checks HDF5 preprocessing version
    2. Defensive dataset loading: Tries multiple possible dataset names
    3. Data quality validation: Validates sampling rate, shape, consistency
    4. Comprehensive error reporting: Clear, actionable error messages
    5. Graceful degradation: Skips bad nights, not entire participants
    
    ATTRIBUTES:
    -----------
    version : str
        Preprocessing version ('v0', 'v1', 'v1t', 'vvt')
    model : torch.nn.Module
        SSL-Wearables HAR-Net10 model
    device : torch.device
        Computation device (CUDA or CPU)
    sampling_rate : int
        Expected sampling rate in Hz (default: 30)
    window_size : int
        Number of samples per window (default: 300 = 10 seconds at 30Hz)
    batch_size : int
        Number of windows to process in parallel (default: 16)
    feature_dim : int
        Dimensionality of extracted features (default: 1024)
    """

    def __init__(self, version='v1'):
        """
        Initialize the robust feature extraction pipeline.
        
        Args:
            version (str): Preprocessing version to use ('v0', 'v1', 'v1t', 'vvt')
        """
        self.version = version
        self.base_dir = Path("/work3/s184484/iRBD-detection")
        
        # ====================================================================
        # VERSION-SPECIFIC PATHS
        # ====================================================================
        self.preprocessed_controls_dir = self.base_dir / "data" / f"preprocessed_{version}" / "controls"
        self.preprocessed_irbd_dir = self.base_dir / "data" / f"preprocessed_{version}" / "irbd"
        
        # Output: Extracted features
        self.features_dir = self.base_dir / "data" / f"features_{version}"
        self.features_controls_dir = self.features_dir / "controls"
        self.features_irbd_dir = self.features_dir / "irbd"
        
        # Logs and visualizations
        self.plots_dir = self.base_dir / "results" / "visualizations" / f"features_{version}"
        self.log_dir = self.base_dir / "validation" / "data_quality_reports"
        
        # Create necessary directories
        for directory in [self.features_controls_dir, self.features_irbd_dir, 
                         self.plots_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # ====================================================================
        # PROCESSING PARAMETERS
        # ====================================================================
        self.sampling_rate = 30  # Hz
        self.window_size = 300   # samples (10 seconds at 30Hz)
        self.batch_size = 16     # windows per batch
        self.feature_dim = 1024  # SSL-Wearables output dimension
        
        # Validation parameters
        self.sampling_rate_tolerance = 1.0  # Hz
        self.min_night_duration = 1.0  # hours
        
        # ====================================================================
        # MODEL SETUP
        # ====================================================================
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ssl_model.to(self.device)
        self.model.eval()
        
        # Debug: Inspect model structure
        print(f"Model device: {self.device}")
        print(f"Batch size: {self.batch_size} windows")
        print(f"Model type: {type(self.model)}")
        
        # Check for feature_extractor
        if hasattr(self.model, 'feature_extractor'):
            print("✓ Model has 'feature_extractor' attribute")
            # Test the feature_extractor with a dummy input
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 300).to(self.device)
                features = self.model.feature_extractor(dummy_input)
                print(f"  Feature extractor output shape: {features.shape}")
                flattened = features.view(1, -1)
                print(f"  Flattened feature dimension: {flattened.shape[-1]}")
        else:
            print("⚠ Model does NOT have 'feature_extractor' attribute")
            print(f"  Available attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')]}")
            raise AttributeError("SSL-Wearables model missing 'feature_extractor' attribute")
        
        # ====================================================================
        # LOGGING SETUP
        # ====================================================================
        log_file = self.log_dir / f"feature_extraction_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"=== ROBUST FEATURE EXTRACTION INITIALIZED ===")
        self.logger.info(f"Version: {version}")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Robustness features enabled:")
        self.logger.info(f"  - Version verification")
        self.logger.info(f"  - Defensive dataset loading")
        self.logger.info(f"  - Data quality validation")
        self.logger.info(f"  - Sampling rate checking (±{self.sampling_rate_tolerance}Hz)")
        
        # ====================================================================
        # STATISTICS TRACKING
        # ====================================================================
        self.stats = {
            'total_participants': 0,
            'processed_participants': 0,
            'failed_participants': 0,
            'controls_processed': 0,
            'irbd_processed': 0,
            'total_nights': 0,
            'skipped_nights': 0,
            'total_windows': 0,
            'total_features_extracted': 0,
            'processing_time_total': 0,
            'version_mismatches': 0,
            'sampling_rate_warnings': 0,
            'data_quality_issues': 0
        }

    def verify_preprocessing_version(self, h5_file):
        """
        Verify that HDF5 file matches expected preprocessing version.
        
        Args:
            h5_file (Path): Path to HDF5 file
            
        Returns:
            tuple: (is_match, actual_version, message)
        """
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'preprocessing_version' in f.attrs:
                    actual_version = f.attrs['preprocessing_version']
                    
                    # Decode if bytes
                    if isinstance(actual_version, bytes):
                        actual_version = actual_version.decode('utf-8')
                    
                    # Version can be 'v1t'
                    # Check if actual version starts with expected version
                    expected = self.version
                    is_match = actual_version.startswith(expected)
                    
                    if is_match:
                        message = f"Version verified: {actual_version}"
                    else:
                        message = (
                            f"VERSION MISMATCH: HDF5 is {actual_version}, "
                            f"but feature extraction expects {expected}"
                        )
                        self.stats['version_mismatches'] += 1
                    
                    return is_match, actual_version, message
                else:
                    # No version attribute - likely older preprocessing
                    message = (
                        f"No preprocessing_version attribute found. "
                        f"Assuming {self.version} (cannot verify)"
                    )
                    return True, None, message
                    
        except Exception as e:
            return False, None, f"Error checking version: {e}"

    def load_dataset_defensive(self, night_group, dataset_type='accel'):
        """
        Defensively load dataset, trying multiple possible names.
        
        Args:
            night_group (h5py.Group): HDF5 group for a night
            dataset_type (str): Type of dataset ('accel' or 'timestamps')
            
        Returns:
            np.ndarray or list: Dataset contents
            
        Raises:
            ValueError: If dataset not found with any expected name
        """
        if dataset_type == 'accel':
            possible_names = ['accel', 'acceleration', 'acc', 'data']
        elif dataset_type == 'timestamps':
            possible_names = ['timestamps', 'timestamp', 'time', 'times']
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Try each possible name
        for name in possible_names:
            if name in night_group:
                data = night_group[name][:]
                
                # Decode bytes to strings for timestamps
                if dataset_type == 'timestamps' and len(data) > 0:
                    if isinstance(data[0], bytes):
                        data = [t.decode('utf-8') for t in data]
                
                self.logger.debug(f"Found {dataset_type} as '{name}'")
                return data
        
        # If we get here, none of the names worked
        available = list(night_group.keys())
        raise ValueError(
            f"Could not find {dataset_type} dataset. "
            f"Tried: {possible_names}. "
            f"Available: {available}"
        )

    def validate_night_data(self, night_number, accel_data, timestamps):
        """
        Validate night data quality and consistency.
        
        Args:
            night_number (int): Night index
            accel_data (np.ndarray): Accelerometer data
            timestamps (list): Timestamp strings
            
        Returns:
            tuple: (is_valid, issues_list)
        """
        issues = []
        
        # Check data consistency
        is_consistent, consistency_msg = validate_data_consistency(accel_data, timestamps)
        if not is_consistent:
            issues.append(f"Consistency: {consistency_msg}")
            self.stats['data_quality_issues'] += 1
        
        # Check sampling rate
        is_valid_rate, actual_hz, rate_msg = validate_sampling_rate(
            timestamps, 
            expected_hz=self.sampling_rate,
            tolerance=self.sampling_rate_tolerance
        )
        if not is_valid_rate and actual_hz is not None:
            issues.append(f"Sampling rate: {rate_msg}")
            self.stats['sampling_rate_warnings'] += 1
        
        # Check minimum duration
        duration_hours = len(accel_data) / (self.sampling_rate * 3600)
        if duration_hours < self.min_night_duration:
            issues.append(
                f"Duration too short: {duration_hours:.1f}h "
                f"(minimum {self.min_night_duration}h)"
            )
        
        # Check if enough data for at least one window
        num_windows = len(accel_data) // self.window_size
        if num_windows == 0:
            issues.append(
                f"Insufficient data: {len(accel_data)} samples < "
                f"{self.window_size} (need at least 1 window)"
            )
        
        return len(issues) == 0, issues

    def load_participant_data(self, h5_file):
        """
        Load preprocessed data with version verification and defensive loading.
        
        Args:
            h5_file (Path): Path to HDF5 file
            
        Returns:
            dict: Participant data with nights
        """
        participant_id = h5_file.stem
        self.logger.info(f"Loading data from: {h5_file.name}")
        
        # ROBUSTNESS: Verify preprocessing version
        is_match, actual_version, version_msg = self.verify_preprocessing_version(h5_file)
        self.logger.info(f"  {version_msg}")
        if not is_match:
            self.logger.warning(f"  Proceeding despite version mismatch")
        
        try:
            with h5py.File(h5_file, 'r') as f:
                nights_data = []
                
                # Find all night groups
                night_keys = [key for key in f.keys() if key.startswith('night')]
                self.logger.info(f"  Found {len(night_keys)} nights")
                
                for night_key in sorted(night_keys):
                    night_group = f[night_key]
                    # Extract night number (e.g., 'night1' -> 1)
                    night_number = int(night_key.replace('night', ''))
                    
                    try:
                        # ROBUSTNESS: Defensive dataset loading
                        acc_data = self.load_dataset_defensive(night_group, 'accel')
                        time_data = self.load_dataset_defensive(night_group, 'timestamps')
                        
                        # ROBUSTNESS: Validate data quality
                        is_valid, issues = self.validate_night_data(
                            night_number, acc_data, time_data
                        )
                        
                        if not is_valid:
                            self.logger.warning(
                                f"  Night {night_number} validation issues: "
                                f"{'; '.join(issues)}"
                            )
                            self.logger.warning(f"  Skipping night {night_number}")
                            self.stats['skipped_nights'] += 1
                            continue
                        
                        nights_data.append({
                            'night_number': night_number,
                            'accel': acc_data,
                            'timestamps': time_data,
                            'num_samples': len(acc_data)
                        })
                        
                        self.logger.info(
                            f"  Night {night_number}: {len(acc_data):,} samples "
                            f"({len(acc_data)/(self.sampling_rate*3600):.1f}h) - OK"
                        )
                        
                    except Exception as e:
                        self.logger.error(
                            f"  Error loading night {night_number}: {e}"
                        )
                        self.logger.warning(f"  Skipping night {night_number}")
                        self.stats['skipped_nights'] += 1
                        continue
                
                if len(nights_data) == 0:
                    raise ValueError(
                        f"No valid nights loaded for {participant_id}. "
                        f"All {len(night_keys)} nights failed validation."
                    )
                
                self.logger.info(
                    f"  Successfully loaded {len(nights_data)}/{len(night_keys)} nights"
                )
                
        except Exception as e:
            self.logger.error(f"Error loading {h5_file.name}: {str(e)}")
            raise
            
        return {
            'participant_id': participant_id,
            'nights': nights_data,
            'num_nights': len(nights_data),
            'preprocessing_version': actual_version
        }

    def find_h5_files(self, directory):
        """Find all .h5 files in a directory."""
        return sorted(list(Path(directory).glob("*.h5")))

    def create_windows(self, acceleration_data):
        """Create non-overlapping 10-second windows from acceleration data."""
        num_samples = len(acceleration_data)
        num_windows = num_samples // self.window_size
        
        if num_windows == 0:
            self.logger.warning(
                f"Insufficient data: {num_samples} samples < "
                f"{self.window_size} (window size)"
            )
            return np.array([])
        
        # Trim to exact number of windows
        trimmed_length = num_windows * self.window_size
        trimmed_data = acceleration_data[:trimmed_length]
        
        # Reshape into windows: (num_windows, window_size, 3)
        windows = trimmed_data.reshape(num_windows, self.window_size, 3)
        
        return windows

    def extract_features_batch(self, windows_batch):
        """
        Extract SSL-Wearables features from a batch of windows.
        
        Args:
            windows_batch (np.ndarray): Windows of shape (batch, 300, 3)
            
        Returns:
            np.ndarray: Features of shape (batch, 1024)
        """
        # Convert to PyTorch tensor and transpose to (batch, channels, time)
        # From: (batch, 300, 3) → To: (batch, 3, 300)
        windows_tensor = torch.from_numpy(windows_batch).float()
        windows_tensor = windows_tensor.permute(0, 2, 1)  # Transpose to (batch, 3, 300)
        windows_tensor = windows_tensor.to(self.device)
        
        # Extract features (no gradient computation needed)
        with torch.no_grad():
            # Check if model has feature_extractor attribute
            if not hasattr(self.model, 'feature_extractor'):
                self.logger.error("Model does not have 'feature_extractor' attribute!")
                self.logger.error(f"Model attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')]}")
                raise AttributeError("SSL-Wearables model missing 'feature_extractor' attribute")
            
            # IMPORTANT: Use feature_extractor, NOT the full model!
            # self.model(x) returns classifier output (batch, class_num=2)
            # self.model.feature_extractor(x) returns features (batch, channels, time)
            # We need to flatten to get (batch, feature_dim) embeddings
            features = self.model.feature_extractor(windows_tensor)
            
            # Flatten features to (batch, feature_dim)
            batch_size = windows_tensor.shape[0]
            embeddings = features.view(batch_size, -1)
            
            # Validate feature dimension
            feature_dim = embeddings.shape[-1]
            if not hasattr(self, '_feature_dim_validated'):
                self.logger.info(f"Feature dimension from model: {feature_dim}")
                if feature_dim != 1024:
                    self.logger.warning(
                        f"Feature dimension is {feature_dim}, not 1024. "
                        f"This is normal for some SSL-Wearables model versions."
                    )
                self._feature_dim_validated = True
            
            # Move back to CPU and convert to numpy
            embeddings = embeddings.cpu().numpy()
        
        return embeddings

    def extract_participant_features(self, participant_data):
        """
        Extract features for all nights of a participant.
        
        Args:
            participant_data (dict): Participant data from load_participant_data
            
        Returns:
            dict: Extracted features and metadata
        """
        participant_id = participant_data['participant_id']
        nights = participant_data['nights']
        
        self.logger.info(f"Extracting features for {participant_id}...")
        
        all_nights_features = []
        all_nights_indices = []
        
        for night_data in nights:
            night_number = night_data['night_number']
            accel = night_data['accel']
            
            # Create windows for this night
            windows = self.create_windows(accel)
            
            if len(windows) == 0:
                self.logger.warning(
                    f"  Night {night_number}: No windows created (insufficient data)"
                )
                continue
            
            # Extract features in batches
            night_features = []
            num_windows = len(windows)
            
            for i in range(0, num_windows, self.batch_size):
                batch = windows[i:i + self.batch_size]
                batch_features = self.extract_features_batch(batch)
                night_features.append(batch_features)
            
            # Concatenate all batches for this night
            night_features = np.concatenate(night_features, axis=0)
            
            # Store features and night indices
            all_nights_features.append(night_features)
            all_nights_indices.extend([night_number] * len(night_features))
            
            self.logger.info(
                f"  Night {night_number}: {len(night_features)} windows → "
                f"{len(night_features)} feature vectors"
            )
            
            self.stats['total_windows'] += len(night_features)
        
        # Validate we have features
        if len(all_nights_features) == 0:
            raise ValueError(f"No features extracted for {participant_id}")
        
        # Keep night-level structure (don't concatenate!)
        # This preserves temporal information for LSTM
        num_nights = len(all_nights_features)
        windows_per_night = [len(night) for night in all_nights_features]
        max_windows = max(windows_per_night)
        total_windows = sum(windows_per_night)
        
        self.logger.info(
            f"  Total: {total_windows} feature vectors from {num_nights} nights"
        )
        self.logger.info(
            f"  Windows per night: min={min(windows_per_night)}, "
            f"max={max_windows}, avg={total_windows/num_nights:.1f}"
        )
        
        self.stats['total_features_extracted'] += total_windows
        self.stats['total_nights'] += num_nights
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'participant_id': participant_id,
            'nights_features': all_nights_features,  # List of (windows, 1024) arrays
            'windows_per_night': windows_per_night,  # List of window counts
            'num_nights': num_nights,
            'max_windows': max_windows,
            'total_windows': total_windows,
            'preprocessing_version': participant_data.get('preprocessing_version')
        }

    def save_features(self, features_dict, output_dir):
        """
        Save extracted features to .npz file in LSTM-compatible format.
        
        Creates 3D array (nights, max_windows, features) with padding and mask.
        
        Args:
            features_dict (dict): Features from extract_participant_features
            output_dir (Path): Output directory
        """
        participant_id = features_dict['participant_id']
        output_file = output_dir / f"{participant_id}.npz"
        
        # Extract night-level features
        nights_features = features_dict['nights_features']  # List of (windows, 1024)
        num_nights = features_dict['num_nights']
        max_windows = features_dict['max_windows']
        feature_dim = nights_features[0].shape[1]  # Should be 1024
        
        # Create 3D array with padding: (nights, max_windows, 1024)
        features_3d = np.zeros((num_nights, max_windows, feature_dim), dtype=np.float32)
        windows_mask = np.zeros((num_nights, max_windows), dtype=bool)
        
        # Fill in actual features and create mask
        for night_idx, night_features in enumerate(nights_features):
            num_windows = len(night_features)
            features_3d[night_idx, :num_windows, :] = night_features
            windows_mask[night_idx, :num_windows] = True  # True = valid data
        
        # Save in LSTM-compatible format
        np.savez_compressed(
            output_file,
            features=features_3d,              # (nights, max_windows, 1024)
            windows_mask=windows_mask,         # (nights, max_windows)
            participant_id=participant_id,
            num_nights=num_nights,
            max_windows=max_windows,
            total_windows=features_dict['total_windows'],
            windows_per_night=np.array(features_dict['windows_per_night']),
            preprocessing_version=features_dict.get('preprocessing_version', 'unknown')
        )
        
        self.logger.info(
            f"Saved features to: {output_file.name} "
            f"(shape: {features_3d.shape}, mask: {windows_mask.sum()}/{windows_mask.size} valid)"
        )

    def test_feature_extraction(self):
        """Test feature extraction with a small sample to verify dimensions."""
        self.logger.info("Testing feature extraction...")
        
        # Create test windows
        test_windows = np.random.randn(2, 300, 3).astype(np.float32)
        
        try:
            features = self.extract_features_batch(test_windows)
            
            # Check dimensions
            if features.shape[0] != 2:
                raise ValueError(f"Expected batch size 2, got {features.shape[0]}")
            
            feature_dim = features.shape[1]
            self.logger.info(f"✓ Feature extraction test successful!")
            self.logger.info(f"  Test input shape: (2, 300, 3)")
            self.logger.info(f"  Feature output shape: {features.shape}")
            self.logger.info(f"  Feature dimension: {feature_dim}")
            
            # Check for NaN or Inf
            if np.any(np.isnan(features)):
                raise ValueError("Features contain NaN values!")
            if np.any(np.isinf(features)):
                raise ValueError("Features contain Inf values!")
            
            self.logger.info(f"  ✓ No NaN or Inf values detected")
            
            # Update expected feature dimension if different
            if feature_dim != self.feature_dim:
                self.logger.warning(
                    f"  Updating expected feature dimension: {self.feature_dim} → {feature_dim}"
                )
                self.feature_dim = feature_dim
                
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Feature extraction test failed: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def process_all_participants(self):
        """Process all participants (controls and iRBD)."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING ROBUST FEATURE EXTRACTION")
        self.logger.info("=" * 80)
        
        # Run a test first
        self.logger.info("\nRunning feature extraction test...")
        if not self.test_feature_extraction():
            self.logger.error("Feature extraction test failed! Aborting.")
            return
        self.logger.info("")
        
        start_time = datetime.now()
        
        # Process controls
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PROCESSING CONTROLS")
        self.logger.info("=" * 80)
        
        control_files = self.find_h5_files(self.preprocessed_controls_dir)
        self.logger.info(f"Found {len(control_files)} control participants")
        
        for h5_file in control_files:
            self.stats['total_participants'] += 1
            try:
                participant_data = self.load_participant_data(h5_file)
                features = self.extract_participant_features(participant_data)
                self.save_features(features, self.features_controls_dir)
                self.stats['processed_participants'] += 1
                self.stats['controls_processed'] += 1
            except Exception as e:
                self.logger.error(f"Failed to process {h5_file.name}: {e}")
                self.logger.error(traceback.format_exc())
                self.stats['failed_participants'] += 1
            
            # Explicit garbage collection
            gc.collect()
        
        # Process iRBD
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PROCESSING iRBD PATIENTS")
        self.logger.info("=" * 80)
        
        irbd_files = self.find_h5_files(self.preprocessed_irbd_dir)
        self.logger.info(f"Found {len(irbd_files)} iRBD participants")
        
        for h5_file in irbd_files:
            self.stats['total_participants'] += 1
            try:
                participant_data = self.load_participant_data(h5_file)
                features = self.extract_participant_features(participant_data)
                self.save_features(features, self.features_irbd_dir)
                self.stats['processed_participants'] += 1
                self.stats['irbd_processed'] += 1
            except Exception as e:
                self.logger.error(f"Failed to process {h5_file.name}: {e}")
                self.logger.error(traceback.format_exc())
                self.stats['failed_participants'] += 1
            
            # Explicit garbage collection
            gc.collect()
        
        # Calculate total time
        end_time = datetime.now()
        self.stats['processing_time_total'] = (end_time - start_time).total_seconds()
        
        # Print final summary
        self.print_summary()

    def print_summary(self):
        """Print final processing summary."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("FEATURE EXTRACTION COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Version: {self.version}")
        self.logger.info(f"Total participants: {self.stats['total_participants']}")
        self.logger.info(f"  Processed: {self.stats['processed_participants']}")
        self.logger.info(f"  Failed: {self.stats['failed_participants']}")
        self.logger.info(f"  Controls: {self.stats['controls_processed']}")
        self.logger.info(f"  iRBD: {self.stats['irbd_processed']}")
        self.logger.info(f"Total nights: {self.stats['total_nights']}")
        self.logger.info(f"  Skipped nights: {self.stats['skipped_nights']}")
        self.logger.info(f"Total windows: {self.stats['total_windows']}")
        self.logger.info(f"Total features: {self.stats['total_features_extracted']}")
        self.logger.info(f"Processing time: {self.stats['processing_time_total']:.1f} seconds")
        self.logger.info(f"\nRobustness statistics:")
        self.logger.info(f"  Version mismatches: {self.stats['version_mismatches']}")
        self.logger.info(f"  Sampling rate warnings: {self.stats['sampling_rate_warnings']}")
        self.logger.info(f"  Data quality issues: {self.stats['data_quality_issues']}")
        self.logger.info("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Robust feature extraction for iRBD detection'
    )
    parser.add_argument(
        '--version',
        type=str,
        required=True,
        choices=['v0', 'v1', 'v1t', 'vvt'],
        help='Preprocessing version to use'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ROBUST FEATURE EXTRACTION FOR IRBD DETECTION")
    print("=" * 80)
    print(f"Version: {args.version}")
    print(f"Robustness features:")
    print(f"  ✓ Version verification")
    print(f"  ✓ Defensive dataset loading")
    print(f"  ✓ Data quality validation")
    print(f"  ✓ Sampling rate checking")
    print("=" * 80)
    
    # Create pipeline and process all participants
    pipeline = RobustFeatureExtractionPipeline(version=args.version)
    pipeline.process_all_participants()
    
    print("\nFeature extraction complete!")
    print(f"Output directory: {pipeline.features_dir}")


if __name__ == "__main__":
    main()