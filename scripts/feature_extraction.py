#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# FEATURE EXTRACTION
# Extract high-level movement features from preprocessed accelerometer data using the SSL-Wearables model.

## INPUT
# Source : .h5 files (from preprocessing stage)
# Directories : 
#    - /work3/s184484/iRBD-detection/data/preprocessing/controls/
#    - /work3/s184484/iRBD-detection/data/preprocessing/irbd/ 
# Format : Clean 30Hz accelerometer data segmented by nights


## PIPELINE
# 1. Night segmentation : Divide each night into 10-minute segments
# 2. Window creation : Create 10-second non-overlapping windows within each segment
# 3. Feature extraction : 
#    - Load SSL-Wearables harnet10 model via torch.hub.load()
#    - Process windows in batches for GPU efficiency
#    - Extract 1024-dimensional features using model.feature_extractor
# 4. Quality control : Validate feature dimensions and handle edge cases
# 5. Memory management : Clear GPU cache between participants
# 6. Save features : Save per-participant features, then auto-combine for LSTM


## OUTPUT
# Format : .npy files
# Directories : 
#    - /work3/s184484/iRBD-detection/data/features/controls/
#    - /work3/s184484/iRBD-detection/data/features/irbd/ 
#    - /work3/s184484/iRBD-detection/data/features/combined/
# Individual files : One .npy file per participant with shape (participant_windows, 1024)
# Combined datasets : 
#    - X_features.npy : All features (total_windows, 1024)
#    - y_labels.npy : All labels (total_windows,) - 0=non-iRBD, 1=iRBD
#    - participant_ids.npy : Participant mapping for sequence organization
# Structure within directories : 
#   ├── features
#   │   ├── controls
#   |   │   ├── participant_PARTICIPANTID0_features.npy     # Shape: (participant_windows, 1024)
#   |   │   ├── participant_PARTICIPANTID1_features.npy     # Shape: (participant_windows, 1024)
#   |   │   └── ...
#   │   ├── irbd
#   |   │   ├── participant_PARTICIPANTID0_features.npy     # Shape: (participant_windows, 1024)
#   |   │   ├── participant_PARTICIPANTID1_features.npy     # Shape: (participant_windows, 1024)
#   |   │   └── ...
#   │   └── combined
#   |       ├── X_features.npy                   # Shape: (total_windows, 1024) - ALL features
#   |       ├── y_labels.npy                     # Shape: (total_windows,) - ALL labels (0=non-iRBD, 1=iRBD)
#   |       ├── participant_ids.npy              # Shape: (total_windows,) - participant mapping
#   |       └── dataset_info.json                # Processing metadata and statistics


## VALIDATION
# Feature dimension consistency : All outputs have exactly 1024 features
# Window count validation : Expected number of windows per night (2880)
# Data integrity : No NaN or infinite values in feature vectors
# Model loading verification : SSL-Wearables model loads correctly
# GPU utilization : Efficient memory usage and processing


## PARAMETERS SUMMARY
# Segment duration : 600 seconds (10 minutes)
# Window size : 300 samples (10 seconds at 30Hz)
# Windows per segment : 60 (non-overlapping)
# Windows per night : 8 hours × 60 minutes × 10 seconds = 2880 windows per night
# Windows per participant : Number of nights (of participant) × 2880
# Feature dimension : 1024 (feature dimensions from SSL-Wearables ResNet output)
# Model : 'harnet10' from 'OxWearables/ssl-wearables'
# Batch size : 8-16 (depeding on GPU memory)
# Input format : (batch, 3, 300) - channels first for SSL-Wearables
# Pre-training : 700,000+ person-days of accelerometer data


## ENVIRONMENT : env_insights


## HPC JOB EXECUTION
# Job script : feature_extraction_job.sh
# Location : /work3/s184484/iRBD-detection/jobs/scripts/feature_extraction_job.sh
# Queue : gpua10 (GPU nodes with shorter queue times)
# Resources : 8 cores, 12GB RAM per core, 1 GPU
# Time limit : 24 hours
# Output logs : /work3/s184484/iRBD-detection/jobs/logs/feature_extraction/feature_extraction_output_JOBID.out
# Error logs : /work3/s184484/iRBD-detection/jobs/logs/feature_extraction/feature_extraction_error_JOBID.err


# In[ ]:


# Basic Python libraries for file operations and system control
import os                    # Operating System interface - helps us work with files and folders
import sys                   # System-specific parameters - helps us control the program execution
import h5py                  # HDF5 library - for reading the preprocessed .h5 files
import numpy as np           # NumPy - for mathematical operations on arrays of numbers
import pandas as pd          # Pandas - for working with data tables and organizing information
from datetime import datetime, timedelta  # For working with dates and times
import logging               # For creating detailed log files that record what the program does
from pathlib import Path     # For easier and more reliable file path handling
import glob                  # For finding files that match specific patterns (like all .h5 files)
import traceback             # For showing detailed error messages when something goes wrong
import json                  # For saving and loading JSON files (configuration and metadata)
import gc                    # Garbage collection - for managing memory usage

# Visualization libraries for creating plots and charts
import matplotlib.pyplot as plt    # Main plotting library - like creating graphs in Excel
import seaborn as sns             # Statistical plotting library - makes beautiful, professional plots

# Configure matplotlib and seaborn for professional-looking plots
plt.style.use('seaborn-v0_8')     # Use seaborn's visual style (makes plots look professional)
sns.set_palette("husl")           # Set a nice color palette (colors that work well together)
plt.rcParams['figure.figsize'] = (12, 8)  # Set default size for all plots (12 inches wide, 8 inches tall)
plt.rcParams['font.size'] = 10    # Set default font size for all text in plots

# Try to import PyTorch and SSL-Wearables - essential for feature extraction
try:
    import torch               # PyTorch - deep learning framework for running SSL-Wearables model
    import torch.nn as nn      # Neural network components from PyTorch
    print(f"PyTorch version: {torch.__version__}")

    # Check if CUDA (GPU) is available for faster processing
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available - will use CPU (slower)")

except ImportError as e:
    # If PyTorch is not installed, show an error message and stop the program
    print(f"Error importing PyTorch: {e}")
    print("Please install PyTorch with: pip install torch")
    sys.exit(1)

# Try to load SSL-Wearables model via torch.hub
try:
    print("Loading SSL-Wearables model...")
    # This downloads and loads the pre-trained harnet10 model
    # SSL-Wearables is trained on over 700,000 days of accelerometer data
    ssl_model = torch.hub.load('OxWearables/ssl-wearables', 'harnet10', pretrained=True, trust_repo=True)
    print("SSL-Wearables model loaded successfully")
except Exception as e:
    print(f"Error loading SSL-Wearables model: {e}")
    print("Please check internet connection and torch.hub access")
    sys.exit(1)


# In[ ]:


# =============================================================================
# FEATURE EXTRACTION PIPELINE CLASS
# =============================================================================

class FeatureExtractionPipeline:
    """
    This class handles the extraction of SSL-Wearables features from preprocessed
    accelerometer data. It organizes features by night (not concatenated) to preserve
    the temporal structure needed for machine learning models.

    WHAT SSL-WEARABLES IS:
    SSL-Wearables is a self-supervised learning model trained on over 700,000 days
    of accelerometer data. It has learned to recognize a wide range of human
    activities and movement patterns, making it perfect for our iRBD detection task.

    WHAT THIS CLASS DOES:
    1. Reads preprocessed .h5 files containing night-segmented accelerometer data
    2. Creates 10-second windows from each night's data separately
    3. Processes windows through SSL-Wearables to extract 1024-dimensional features
    4. Organizes features by night: (nights, windows_per_night, 1024_features)
    5. Saves structured feature arrays suitable for machine learning models

    WHY NIGHT-BASED ORGANIZATION:
    - Preserves temporal structure of sleep data
    - Allows models to learn night-to-night patterns
    - Enables proper cross-validation (participant-level splits)
    - Supports variable-length sequences (different numbers of nights per participant)
    """

    def __init__(self):
        """
        Initialize the feature extraction pipeline with all necessary configuration.
        This sets up directories, parameters, and the SSL-Wearables model.
        """

        # =================================================================
        # CONFIGURATION SECTION - Choose between example test and full processing
        # =================================================================

        # FOR EXAMPLE FILE TESTING
        self.base_dir = Path("/work3/s184484/iRBD-detection")  # Main project folder on HPC
        #self.mode = "EXAMPLE_TEST"                             # Tell the script we're testing with example
        self.example_participant = "2290025_90001_0_0"        # Example participant ID (without .h5 extension)

        # FOR FULL DATASET PROCESSING 
        self.mode = "FULL_PROCESSING"                          # Tell the script we're processing all files

        # =================================================================
        # DIRECTORY SETUP - Define where to find files and save results
        # =================================================================

        # Input directories (where the preprocessed .h5 files are stored):
        self.preprocessed_controls_dir = self.base_dir / "data" / "preprocessed" / "controls"  # Healthy people's data
        self.preprocessed_irbd_dir = self.base_dir / "data" / "preprocessed" / "irbd"          # iRBD patients' data

        # Output directories (where to save the extracted features):
        self.features_dir = self.base_dir / "data" / "features"                            # Main features directory
        self.features_controls_dir = self.features_dir / "controls"                       # Individual control features
        self.features_irbd_dir = self.features_dir / "irbd"                               # Individual iRBD features
        self.features_combined_dir = self.features_dir / "combined"                       # Combined training datasets

        # Visualization output directory (where to save plots for the report):
        self.plots_dir = self.base_dir / "results" / "visualizations"
        self.example_plots_dir = self.plots_dir / "example_testing"  # Specific folder for example file plots

        # Only create log directory if it doesn't exist (for logging only)
        self.log_dir = self.base_dir / "validation" / "data_quality_reports"
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # =================================================================
        # PROCESSING PARAMETERS - Settings that control feature extraction
        # =================================================================

        # Windowing parameters (how we split continuous data into chunks):
        self.window_size_seconds = 10           # Each window is 10 seconds long
        self.sampling_rate = 30                 # Data is sampled at 30Hz (30 samples per second)
        self.window_size_samples = self.window_size_seconds * self.sampling_rate  # 300 samples per window
        self.window_overlap = 0.0               # No overlap between windows (0% overlap)

        # Expected windows per night calculation:
        # 8 hours × 60 minutes/hour × 60 seconds/minute = 28,800 seconds per night
        # 28,800 seconds ÷ 10 seconds/window = 2,880 windows per perfect 8-hour night
        self.expected_windows_per_night = 8 * 60 * 60 // self.window_size_seconds  # 2880 windows

        # WHY 10-SECOND WINDOWS:
        # 10 seconds captures meaningful movement patterns while providing enough
        # temporal resolution for detecting brief iRBD episodes during sleep.

        # Model parameters (settings for SSL-Wearables processing):
        self.feature_dim = 1024                 # SSL-Wearables outputs 1024-dimensional feature vectors
        self.batch_size = 512                   # Process 512 windows at once (GPU memory dependent)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

        # WHY BATCH PROCESSING:
        # Processing windows in batches is much faster than one-by-one processing
        # and makes efficient use of GPU memory.

        # Visualization parameters (control which plots to create):
        self.create_individual_plots = True     # Create detailed plots for each participant
        self.create_summary_plots = True       # Create overall summary plots

        # =================================================================
        # MODEL SETUP - Prepare SSL-Wearables for feature extraction
        # =================================================================

        # Move the SSL-Wearables model to GPU (if available) for faster processing
        self.ssl_model = ssl_model.to(self.device)

        # Set the model to evaluation mode (no training, just inference)
        self.ssl_model.eval()

        # Disable gradient computation (saves memory and speeds up processing)
        torch.set_grad_enabled(False)

        # Log model information
        print(f"SSL-Wearables model ready on device: {self.device}")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Window size: {self.window_size_seconds}s ({self.window_size_samples} samples)")
        print(f"Expected windows per night: ~{self.expected_windows_per_night}")

        # Initialize the supporting systems
        self.setup_logging()                # Set up the system to record what happens
        self.initialize_stats()             # Set up counters to track our progress

        # Print information about what mode we're running in
        print(f"Running in {self.mode} mode")
        print(f"Base directory: {self.base_dir}")

    def initialize_stats(self):
        """
        Set up counters to keep track of processing statistics.
        This helps us monitor success rates and identify any problems.
        """
        self.stats = {
            'total_participants': 0,        # How many participant files we found
            'processed_participants': 0,    # How many participants we successfully processed
            'failed_participants': 0,       # How many participants had errors
            'total_nights': 0,              # Total number of nights across all participants
            'total_windows': 0,             # Total number of 10-second windows processed
            'total_features_extracted': 0,  # Total number of feature vectors created
            'controls_processed': 0,        # How many control participants processed
            'irbd_processed': 0,            # How many iRBD participants processed
            'processing_time_total': 0.0    # Total time spent on feature extraction
        }

    def setup_logging(self):
        """
        Set up the logging system to record everything that happens during feature extraction.
        This creates a detailed record of the process for debugging and documentation.
        """
        # Create a unique log file name with current date and time
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"feature_extraction_{self.mode.lower()}_{current_time}.log"

        # Configure the logging system
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),      # Save messages to log file
                logging.StreamHandler(sys.stdout)   # Also display on screen
            ]
        )

        # Create our logger object
        self.logger = logging.getLogger(__name__)

        # Write initial log messages
        self.logger.info(f"=== Feature Extraction Pipeline Started ({self.mode}) ===")
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Window size: {self.window_size_seconds}s ({self.window_size_samples} samples)")
        self.logger.info(f"Expected windows per night: ~{self.expected_windows_per_night}")
        self.logger.info(f"Batch size: {self.batch_size}")

    def find_h5_files(self, directory):
        """
        Look for all .h5 files in a specific directory.
        These are the preprocessed files created by the preprocessing pipeline.

        Args:
            directory: The folder path where we want to search for .h5 files

        Returns:
            A list of file paths for all .h5 files found in the directory
        """
        # Search for .h5 files in the directory
        h5_pattern = directory / "*.h5"
        h5_files = glob.glob(str(h5_pattern))

        # Convert file paths from strings to Path objects
        h5_files = [Path(f) for f in h5_files]

        # Sort files alphabetically for consistent processing order
        h5_files.sort()

        # Log how many files we found
        self.logger.info(f"Found {len(h5_files)} .h5 files in {directory}")

        return h5_files

    def load_participant_data(self, h5_file):
        """
        Load preprocessed data from an .h5 file for one participant.
        This reads all nights separately to preserve the night structure.

        IMPORTANT: Unlike the previous version, this function keeps nights SEPARATE
        instead of concatenating them. This preserves the temporal structure needed
        for machine learning models.

        Args:
            h5_file: Path to the .h5 file to read

        Returns:
            dict: Dictionary containing participant data organized by night
        """
        try:
            # Extract participant ID from filename (remove .h5 extension)
            participant_id = h5_file.stem

            self.logger.info(f"Loading participant: {participant_id}")

            # Open the HDF5 file for reading
            with h5py.File(h5_file, 'r') as f:
                # Read participant information from file attributes
                participant_name = f.attrs['name']
                num_nights = f.attrs['number_of_nights']

                self.logger.info(f"   - Participant: {participant_name}")
                self.logger.info(f"   - Number of nights: {num_nights}")

                # Initialize list to store data for each night SEPARATELY
                nights_data = []  # Each element will be one night's data

                # Loop through all nights and collect each night's data separately
                for night_num in range(1, num_nights + 1):
                    night_group_name = f"night{night_num}"

                    # Check if this night group exists in the file
                    if night_group_name in f:
                        night_group = f[night_group_name]

                        # Read accelerometer data for this specific night
                        x_data = night_group['x'][:]  # X-axis data
                        y_data = night_group['y'][:]  # Y-axis data
                        z_data = night_group['z'][:]  # Z-axis data

                        # Read timestamps (stored as ISO format strings)
                        timestamps_str = night_group['timestamps'][:]
                        # Convert string timestamps back to datetime objects
                        timestamps = [pd.Timestamp(ts.decode('utf-8')) for ts in timestamps_str]

                        # Combine x, y, z data for this night
                        night_data = np.column_stack([x_data, y_data, z_data])

                        # Store this night's data as a separate entry
                        nights_data.append({
                            'night_number': night_num,
                            'data': night_data,              # Shape: (samples_this_night, 3)
                            'timestamps': timestamps,
                            'samples': len(night_data),
                            'duration_hours': len(night_data) / self.sampling_rate / 3600
                        })

                        self.logger.info(f"     - Night {night_num}: {len(night_data):,} samples "
                                       f"({len(night_data) / self.sampling_rate / 3600:.1f}h)")
                    else:
                        self.logger.warning(f"     - Night {night_num}: Group not found")

                # Calculate total statistics
                total_samples = sum(night['samples'] for night in nights_data)
                total_duration_hours = sum(night['duration_hours'] for night in nights_data)

                self.logger.info(f"   - Total samples across all nights: {total_samples:,}")
                self.logger.info(f"   - Total duration: {total_duration_hours:.1f} hours")
                self.logger.info(f"   - Valid nights loaded: {len(nights_data)}")

                # Return all the participant information with nights kept separate
                return {
                    'participant_id': participant_id,
                    'participant_name': participant_name,
                    'num_nights': len(nights_data),  # Actual number of valid nights loaded
                    'nights_data': nights_data,      # List of night data (NOT concatenated)
                    'total_samples': total_samples,
                    'total_duration_hours': total_duration_hours
                }

        except Exception as e:
            # If anything goes wrong, log the error and re-raise it
            self.logger.error(f"Error loading {h5_file.name}: {str(e)}")
            raise

    def create_windows_for_night(self, night_data):
        """
        Split one night's accelerometer data into fixed-size windows for feature extraction.
        Each window contains 10 seconds of data (300 samples at 30Hz).

        This function processes ONE NIGHT at a time, preserving the night structure.

        Args:
            night_data: Accelerometer data for one night (samples × 3)

        Returns:
            numpy array: Windowed data for this night (num_windows × window_size × 3)
        """
        total_samples = len(night_data)

        # Calculate how many complete windows we can create from this night
        num_windows = total_samples // self.window_size_samples

        # Calculate how many samples we'll actually use (might be slightly less than total)
        samples_used = num_windows * self.window_size_samples

        # ERROR CHECK: Make sure we have at least one complete window
        if num_windows == 0:
            raise ValueError(f"Not enough data in this night to create even one window. "
                           f"Need at least {self.window_size_samples} samples, got {total_samples}.")

        # Reshape the data into windows
        # Take only the samples that fit into complete windows
        data_for_windowing = night_data[:samples_used]

        # Reshape into windows: (num_windows, window_size_samples, 3)
        windows = data_for_windowing.reshape(num_windows, self.window_size_samples, 3)

        return windows

    def extract_features_batch(self, windows_batch):
        """
        Extract SSL-Wearables features from a batch of windows.
        This is where the actual feature extraction happens using the pre-trained model.

        Args:
            windows_batch: Batch of windows (batch_size × window_size × 3)

        Returns:
            numpy array: Feature vectors (batch_size × 1024)
        """
        try:
            # Convert numpy array to PyTorch tensor
            # SSL-Wearables expects input in format: (batch, channels, samples)
            # Our windows are in format: (batch, samples, channels)
            # So we need to swap the last two dimensions
            windows_tensor = torch.tensor(windows_batch, dtype=torch.float32).permute(0, 2, 1)

            # Move tensor to GPU (if available) for faster processing
            windows_tensor = windows_tensor.to(self.device)

            # Extract features using SSL-Wearables feature extractor
            with torch.no_grad():  # Disable gradient computation for efficiency
                features = self.ssl_model.feature_extractor(windows_tensor).squeeze(-1)
                feature_batch = features.detach().cpu().numpy()

            return feature_batch

        except Exception as e:
            self.logger.error(f"Error in feature extraction: {str(e)}")
            raise

    def extract_features_for_night(self, night_data):
        """
        Extract SSL-Wearables features for one complete night.
        This processes all windows from one night and returns the features.

        Args:
            night_data: Dictionary containing one night's accelerometer data

        Returns:
            numpy array: Feature vectors for this night (num_windows × 1024)
        """
        try:
            night_num = night_data['night_number']
            data = night_data['data']

            self.logger.info(f"   Extracting features for night {night_num}")

            # STEP 1: Create windows from this night's data
            windows = self.create_windows_for_night(data)
            num_windows = len(windows)

            self.logger.info(f"       - Windows created: {num_windows:,} "
                           f"(expected ~{self.expected_windows_per_night})")

            # STEP 2: Process windows in batches for efficiency
            all_features = []  # List to store feature vectors from all batches

            # Calculate how many batches we need
            num_batches = (num_windows + self.batch_size - 1) // self.batch_size  # Ceiling division

            # Process each batch
            for batch_idx in range(num_batches):
                # Calculate start and end indices for this batch
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_windows)

                # Extract windows for this batch
                batch_windows = windows[start_idx:end_idx]

                # Extract features for this batch
                batch_features = self.extract_features_batch(batch_windows)

                # Add to our collection of all features
                all_features.append(batch_features)

                # Clear GPU cache to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # STEP 3: Combine all batch results for this night
            night_features = np.concatenate(all_features, axis=0)

            self.logger.info(f"       - Features extracted: {night_features.shape}")

            return night_features

        except Exception as e:
            self.logger.error(f"Error extracting features for night {night_num}: {str(e)}")
            raise

    def extract_participant_features(self, participant_data):
        """
        Extract SSL-Wearables features for one complete participant.
        This processes all nights for one participant and organizes features by night.

        IMPORTANT: This function maintains the night structure:
        Output shape: (nights, windows_per_night, 1024_features)

        Args:
            participant_data: Dictionary containing participant's accelerometer data organized by night

        Returns:
            dict: Dictionary containing extracted features organized by night
        """
        try:
            participant_id = participant_data['participant_id']
            nights_data = participant_data['nights_data']

            # Record start time for performance tracking
            start_time = datetime.now()

            self.logger.info(f"Extracting features for {participant_id} ({len(nights_data)} nights)")

            # Initialize list to store features for each night
            all_nights_features = []  # Each element will be features for one night
            total_windows = 0

            # Process each night separately
            for night_data in nights_data:
                # Extract features for this specific night
                night_features = self.extract_features_for_night(night_data)

                # Add this night's features to our collection
                all_nights_features.append(night_features)
                total_windows += len(night_features)

            # Convert list of night features to a structured numpy array
            # We need to handle the fact that different nights might have different numbers of windows

            # Find the maximum number of windows across all nights
            max_windows_per_night = max(len(night_features) for night_features in all_nights_features)

            # Create a padded array to store all nights' features
            # Shape: (num_nights, max_windows_per_night, 1024)
            # We'll pad shorter nights with zeros
            num_nights = len(all_nights_features)
            participant_features = np.zeros((num_nights, max_windows_per_night, self.feature_dim))

            # Fill in the actual features for each night
            for night_idx, night_features in enumerate(all_nights_features):
                num_windows_this_night = len(night_features)
                participant_features[night_idx, :num_windows_this_night, :] = night_features

            # Also create a mask to indicate which windows are real vs padded
            windows_mask = np.zeros((num_nights, max_windows_per_night), dtype=bool)
            for night_idx, night_features in enumerate(all_nights_features):
                num_windows_this_night = len(night_features)
                windows_mask[night_idx, :num_windows_this_night] = True

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Log final results
            self.logger.info(f" Feature extraction completed:")
            self.logger.info(f"   - Nights processed: {num_nights}")
            self.logger.info(f"   - Total windows: {total_windows:,}")
            self.logger.info(f"   - Features shape: {participant_features.shape}")
            self.logger.info(f"   - Max windows per night: {max_windows_per_night}")
            self.logger.info(f"   - Processing time: {processing_time:.1f} seconds")
            self.logger.info(f"   - Speed: {total_windows/processing_time:.1f} windows/second")

            # Return all the results with night structure preserved
            return {
                'participant_id': participant_id,
                'features': participant_features,       # Shape: (nights, max_windows_per_night, 1024)
                'windows_mask': windows_mask,           # Shape: (nights, max_windows_per_night) - True for real windows
                'num_nights': num_nights,
                'total_windows': total_windows,
                'max_windows_per_night': max_windows_per_night,
                'windows_per_night': [len(night_features) for night_features in all_nights_features],
                'processing_time': processing_time,
                'windows_per_second': total_windows / processing_time
            }

        except Exception as e:
            self.logger.error(f"Error extracting features for {participant_id}: {str(e)}")
            raise

    def save_participant_features(self, feature_data, output_path):
        """
        Save extracted features for one participant to a .npy file.
        The features are saved with night structure preserved.

        Args:
            feature_data: Dictionary containing participant features and metadata
            output_path: Full path where to save the feature file
        """
        try:
            participant_id = feature_data['participant_id']
            features = feature_data['features']
            windows_mask = feature_data['windows_mask']

            # Create a dictionary with all the data we want to save
            save_data = {
                'features': features,                    # Shape: (nights, max_windows_per_night, 1024)
                'windows_mask': windows_mask,            # Shape: (nights, max_windows_per_night)
                'participant_id': participant_id,
                'num_nights': feature_data['num_nights'],
                'total_windows': feature_data['total_windows'],
                'max_windows_per_night': feature_data['max_windows_per_night'],
                'windows_per_night': feature_data['windows_per_night'],
                'extraction_date': datetime.now().isoformat()
            }

            # Save as numpy file (can store dictionaries)
            np.save(output_path, save_data)

            # Log successful save
            self.logger.info(f" Saved features: {output_path.name}")
            self.logger.info(f"   - Shape: {features.shape}")
            self.logger.info(f"   - File size: {output_path.stat().st_size / 1e6:.1f} MB")

        except Exception as e:
            self.logger.error(f"Error saving features for {feature_data['participant_id']}: {str(e)}")
            raise

    def process_participant(self, h5_file, group_type):
        """
        Process one participant from start to finish.
        This is the main processing function that orchestrates all steps for one person.

        Args:
            h5_file: Path to the participant's .h5 file
            group_type: Either 'controls' or 'irbd'
        """
        try:
            # Extract participant ID from filename
            participant_id = h5_file.stem

            # Log that we're starting to process this participant
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing {group_type.upper()}: {participant_id}")
            self.logger.info(f"{'='*60}")

            # STEP 1: Load participant data from .h5 file (keeping nights separate)
            participant_data = self.load_participant_data(h5_file)

            # STEP 2: Extract SSL-Wearables features (preserving night structure)
            feature_data = self.extract_participant_features(participant_data)

            # STEP 3: Save features to appropriate location
            if self.mode == "EXAMPLE_TEST":
                # For example testing, save to root of features directory
                output_path = self.features_dir / "example_file_features.npy"
            else:
                # For full processing, save to appropriate group directory
                if group_type == 'controls':
                    output_dir = self.features_controls_dir
                else:
                    output_dir = self.features_irbd_dir
                output_path = output_dir / f"{participant_id}_features.npy"

            self.save_participant_features(feature_data, output_path)

            # STEP 4: Update statistics
            self.stats['processed_participants'] += 1
            self.stats['total_nights'] += feature_data['num_nights']
            self.stats['total_windows'] += feature_data['total_windows']
            self.stats['total_features_extracted'] += feature_data['total_windows']
            self.stats['processing_time_total'] += feature_data['processing_time']

            if group_type == 'controls':
                self.stats['controls_processed'] += 1
            else:
                self.stats['irbd_processed'] += 1

            # Log success
            self.logger.info(f"{participant_id} processed successfully:")
            self.logger.info(f"   - {feature_data['num_nights']} nights")
            self.logger.info(f"   - {feature_data['total_windows']:,} windows")
            self.logger.info(f"   - Features shape: {feature_data['features'].shape}")

        except Exception as e:
            # If anything goes wrong, log the error but continue with other participants
            self.logger.error(f"Error processing {participant_id}: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.stats['failed_participants'] += 1

    def run_example_test(self):
        """
        Run feature extraction on just one example participant for testing.
        This saves the result as 'example_file_features.npy' in the root features directory.
        """
        self.logger.info("EXAMPLE TEST MODE: Processing example participant")

        # Look for the example participant in both directories
        example_file = None
        group_type = None

        # Check controls directory first
        controls_file = self.preprocessed_controls_dir / f"{self.example_participant}.h5"
        if controls_file.exists():
            example_file = controls_file
            group_type = 'controls'
        else:
            # Check iRBD directory
            irbd_file = self.preprocessed_irbd_dir / f"{self.example_participant}.h5"
            if irbd_file.exists():
                example_file = irbd_file
                group_type = 'irbd'

        # Check if we found the example file
        if example_file is None:
            self.logger.error(f"Example participant not found: {self.example_participant}")
            self.logger.error(f"Checked: {controls_file}")
            self.logger.error(f"Checked: {irbd_file}")
            return

        # Set statistics for processing one participant
        self.stats['total_participants'] = 1

        # Process the example participant
        self.process_participant(example_file, group_type)

        # Show final results
        self.print_final_statistics()

        # Report success or failure
        if self.stats['processed_participants'] > 0:
            self.logger.info("EXAMPLE TEST SUCCESSFUL!")
            self.logger.info(f"Features saved to: {self.features_dir}/example_file_features.npy")
            self.logger.info("Ready for full dataset processing!")
        else:
            self.logger.error("EXAMPLE TEST FAILED!")

    def run_full_processing(self):
        """
        Run feature extraction on all participants in the dataset.
        This processes all .h5 files in both controls and iRBD directories.
        """
        self.logger.info("FULL PROCESSING MODE: Processing all participants")

        # Find all .h5 files in both directories
        controls_files = self.find_h5_files(self.preprocessed_controls_dir)
        irbd_files = self.find_h5_files(self.preprocessed_irbd_dir)

        # Calculate total number of participants
        total_participants = len(controls_files) + len(irbd_files)
        self.stats['total_participants'] = total_participants

        # Check if we found any files
        if total_participants == 0:
            self.logger.error("No .h5 files found in input directories")
            return

        # Log what we found
        self.logger.info(f"Found {total_participants} participants:")
        self.logger.info(f"   - Controls: {len(controls_files)} participants")
        self.logger.info(f"   - iRBD: {len(irbd_files)} participants")

        # Process all control participants
        self.logger.info(f"\n Processing CONTROLS ({len(controls_files)} participants)...")
        for i, h5_file in enumerate(controls_files, 1):
            self.logger.info(f"\n--- Controls Progress: {i}/{len(controls_files)} ---")

            # CHECK IF ALREADY PROCESSED (RESUME FUNCTIONALITY)
            participant_id = h5_file.stem
            feature_file = self.features_controls_dir / f"{participant_id}_features.npy"
            if feature_file.exists():
                self.logger.info(f"✓ Skipping {participant_id} - features already exist")
                self.stats['processed_participants'] += 1
                self.stats['controls_processed'] += 1
                continue

            self.process_participant(h5_file, 'controls')

        # Process all iRBD participants
        self.logger.info(f"\n Processing iRBD ({len(irbd_files)} participants)...")
        for i, h5_file in enumerate(irbd_files, 1):
            self.logger.info(f"\n--- iRBD Progress: {i}/{len(irbd_files)} ---")

            # CHECK IF ALREADY PROCESSED (RESUME FUNCTIONALITY)
            participant_id = h5_file.stem
            feature_file = self.features_irbd_dir / f"{participant_id}_features.npy"
            if feature_file.exists():
                self.logger.info(f"✓ Skipping {participant_id} - features already exist")
                self.stats['processed_participants'] += 1
                self.stats['irbd_processed'] += 1
                continue

            self.process_participant(h5_file, 'irbd')

        # Show final statistics
        self.print_final_statistics()

    def run_feature_extraction(self):
        """
        Main function to run the feature extraction pipeline.
        Decides whether to run example test or full processing based on configuration.
        """
        if self.mode == "EXAMPLE_TEST":
            self.run_example_test()
        else:
            self.run_full_processing()

    def print_final_statistics(self):
        """
        Print comprehensive summary of the feature extraction results.
        Shows processing success rates, performance metrics, and dataset statistics.
        """
        # Print header
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"FEATURE EXTRACTION COMPLETED ({self.mode})")
        self.logger.info(f"{'='*60}")

        # Participant processing statistics
        self.logger.info(f"Participant Statistics:")
        self.logger.info(f"  - Total participants: {self.stats['total_participants']}")
        self.logger.info(f"  - Successfully processed: {self.stats['processed_participants']}")
        self.logger.info(f"  - Failed to process: {self.stats['failed_participants']}")

        # Calculate success rate
        if self.stats['total_participants'] > 0:
            success_rate = self.stats['processed_participants'] / self.stats['total_participants'] * 100
            self.logger.info(f"  - Success rate: {success_rate:.1f}%")

        self.logger.info("")

        # Group breakdown
        self.logger.info(f"Group Breakdown:")
        self.logger.info(f"  - Controls processed: {self.stats['controls_processed']}")
        self.logger.info(f"  - iRBD processed: {self.stats['irbd_processed']}")
        self.logger.info("")

        # Data processing statistics
        self.logger.info(f"Data Processing:")
        self.logger.info(f"  - Total nights: {self.stats['total_nights']}")
        self.logger.info(f"  - Total windows: {self.stats['total_windows']:,}")
        self.logger.info(f"  - Total features extracted: {self.stats['total_features_extracted']:,}")

        # Calculate averages
        if self.stats['processed_participants'] > 0:
            avg_nights = self.stats['total_nights'] / self.stats['processed_participants']
            avg_windows = self.stats['total_windows'] / self.stats['processed_participants']
            self.logger.info(f"  - Average nights per participant: {avg_nights:.1f}")
            self.logger.info(f"  - Average windows per participant: {avg_windows:.0f}")

        self.logger.info("")

        # Performance statistics
        self.logger.info(f"Performance:")
        self.logger.info(f"  - Total processing time: {self.stats['processing_time_total']:.1f} seconds")

        if self.stats['processing_time_total'] > 0:
            windows_per_second = self.stats['total_windows'] / self.stats['processing_time_total']
            self.logger.info(f"  - Overall speed: {windows_per_second:.1f} windows/second")

        self.logger.info("")

        # Final success message
        if self.stats['processed_participants'] > 0:
            self.logger.info("Feature extraction pipeline completed successfully!")
            if self.mode == "EXAMPLE_TEST":
                self.logger.info(f"Example features saved to: /data/features/example_file_features.npy")
        else:
            self.logger.error("Feature extraction pipeline failed!")


# In[ ]:


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """
    Main function that runs when the script is executed directly.
    Creates a FeatureExtractionPipeline object and runs the entire process.
    """
    try:
        # Create and run the feature extraction pipeline
        pipeline = FeatureExtractionPipeline()
        pipeline.run_feature_extraction()

    except KeyboardInterrupt:
        print("\n Feature extraction interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n Feature extraction failed with error: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

# =============================================================================
# SCRIPT EXECUTION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()


# In[ ]:


### FOR THE EXAMPLE FILE
# 7 nights processed
# 864000 samples per night (8hours at 30 Hz)
# 48 segments per night (10-minute segments)
# 60 windows per segment (10-second windows)
# 2880 windows per night total

# Each segment: (60, 1024) features
# Each night: (2880, 1024) features
# Total: 7 x 2880 = 20160 windows with 1024D features

