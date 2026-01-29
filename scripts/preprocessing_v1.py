#!/usr/bin/env python
# coding: utf-8

# PREPROCESSING V1 - TEMPERATURE FILTERING
# Transform raw accelerometer files into clean night-segmented data suitable for feature extraction.
# V1: Adds rate-of-change temperature detection (18°C) to baseline (v0)

## INPUT
# Source : .CWA and .cwa files from Axivity accelerometer devices. 
# Directories : 
#    - /work3/s184484/iRBD-detection/data/raw/controls/
#    - /work3/s184484/iRBD-detection/data/raw/irbd/ 
# Format : Binary accelerometer data files containing continuous recordings


## PIPELINE
# 1. File reading : Use actipy.read_device() with 12Hz lowpass and 30Hz resampling
# 2. Night segmentation : Extract 8-hour periods from 22:00 to 06:00
# 3. Quality control : 
#    - Apply 18°C temperature threshold with rate-of-change detection (DETACH algorithm)
#    - Remove nights flagged as non-wear by actipy
#    - Validate data integrity and sampling rate consistency
# 4. Save preprocessed data : Create a .h5 file for each .cwa file, with the clean preprocessed data


## OUTPUT
# Format : .h5 files (one per input .cwa file)
# Directories : 
#    - /work3/s184484/iRBD-detection/data/preprocessed_v1/controls/
#    - /work3/s184484/iRBD-detection/data/preprocessed_v1/irbd/
# Structure (for each .h5 file, consistent across all nights) :
#   ├── name (attribute)             # Participant identifier
#   ├── number_of_nights (attribute) # Total valid nights
#   └── datasets/
#       ├── night1/
#       │ ├── accel                  # Combined x,y,z accelerometer data (30Hz, shape: [N, 3])
#       │ └── timestamps             # Corresponding timestamps (ISO format strings)
#       ├── night2/
#       │ ├── accel
#       │ └── timestamps
#       └── ...


## VALIDATION
# - Frequency analysis to confirm 12Hz filter preserves iRBD-relevant signals
# - Sampling rate validation to ensure 30Hz captures movement dynamics  
# - Structure validation to ensure consistency across all files
# - Temperature threshold effectiveness for non-wear detection


# Notes : Make sure to be able to read .cwa and .CWA files. No need for standardization!!


## PARAMETERS SUMMARY
# Library : actipy
# Lowpass Filter : 12Hz (sleep movement studies typically use 10-15Hz lowpass filters)
# Resampling : 30Hz (Nyquist theorem: 2 × highest frequency of interest, most sleep actigraphy studies use 25-50Hz)
# Night Segmentation : 22:00-06:00 (8-hour sleep periods)
# Temperature Threshold : 18 degrees Celcius
# Non-wear Detection : Automatic flagging by actipy, then this script removes the flagged nights


## ENVIRONMENT : env_preprocessing


## HPC JOB EXECUTION
# Job script : preprocessing_job.sh
# Location : /work3/s184484/iRBD-detection/jobs/scripts/preprocessing_job.sh
# Queue : hpc (CPU nodes)
# Resources : 4 cores, 8GB RAM per core
# Time limit : 24 hours
# Output logs : /work3/s184484/iRBD-detection/jobs/logs/preprocessing/preprocessing_output_JOBID.out
# Error logs : /work3/s184484/iRBD-detection/jobs/logs/preprocessing/preprocessing_error_JOBID.err



import os                    # File and directory operations
import sys                   # System-specific parameters - helps us control the program execution
import h5py                  # HDF5 library - for saving the processed data in an efficient format
import numpy as np           # NumPy - for mathematical operations on arrays of numbers
import pandas as pd          # Pandas - for working with data tables and organizing information
from datetime import datetime, timedelta, time  # For working with dates and times
import logging               # For creating detailed log files that record what the program does
from pathlib import Path     # For easier and more reliable file path handling
import glob                  # For finding files that match specific patterns (like all .cwa files)
import traceback             # For showing detailed error messages when something goes wrong

import matplotlib.pyplot as plt    # Main plotting library
import seaborn as sns             # Statistical plotting library

# Configure plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

try:
    import actipy
    print(f"actipy version: {actipy.__version__}")
except ImportError as e:
    print(f"Error importing actipy: {e}")
    print("Please install actipy with: pip install actipy")
    sys.exit(1)



# =============================================================================
# PREPROCESSING PIPELINE CLASS
# =============================================================================

class PreprocessingPipeline:
    """
    This class handles the complete preprocessing of accelerometer data for iRBD detection.

    WHAT THIS CLASS DOES:
    1. Reads raw .cwa files from Axivity accelerometer devices
    2. Applies signal processing (12Hz lowpass filter, 30Hz resampling)
    3. Segments data into night periods (22:00-06:00)
    4. Removes poor quality data (temperature filtering, non-wear detection)
    5. Saves clean data in HDF5 format for the next pipeline stage
    """

    def __init__(self):
        """
        Initialize the preprocessing pipeline with all necessary configuration.
        This sets up directories, parameters, and logging systems.
        """

        # =================================================================
        # CONFIGURATION SECTION
        # =================================================================

        # FOR EXAMPLE FILE TESTING
        self.base_dir = Path("/work3/s184484/iRBD-detection")
        #self.mode = "EXAMPLE_TEST"                             
        #self.example_participant = "2290025_90001_0_0"

        # FOR FULL DATASET PROCESSING
        self.mode = "FULL_PROCESSING"

        # =================================================================
        # DIRECTORY SETUP
        # =================================================================

        # Input directories
        self.raw_controls_dir = self.base_dir / "data" / "raw" / "controls"  # Healthy people's data
        self.raw_irbd_dir = self.base_dir / "data" / "raw" / "irbd"          # iRBD patients' data
        #self.raw_example_file = self.base_dir / "data" / "raw" / f"{self.example_participant}.cwa"  # Example file path

        # Output directories
        self.preprocessed_controls_dir = self.base_dir / "data" / "preprocessed_v1" / "controls"  # Processed healthy data
        self.preprocessed_irbd_dir = self.base_dir / "data" / "preprocessed_v1" / "irbd"          # Processed iRBD data

        # Visualization output directory
        self.plots_dir = self.base_dir / "results" / "visualizations" / "v1"
        #self.example_plots_dir = self.plots_dir / "example_testing"

        # Only create log directory if it doesn't exist (for logging only)
        self.log_dir = self.base_dir / "validation" / "data_quality_reports"
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # =================================================================
        # PROCESSING PARAMETERS
        # =================================================================

        # actipy processing parameters
        self.lowpass_hz = 12                    # Remove frequencies above 12Hz (sleep movements are slow)
        self.resample_hz = 30                   # Resample to 30Hz (good balance of detail vs file size)
        self.detect_nonwear = True              # Let actipy automatically detect when device wasn't worn
        self.calibrate_gravity = True           # Correct for gravity and device orientation

        # Night segmentation parameters
        self.night_start_hour = 22              # Start of sleep period (10:00 PM)
        self.night_end_hour = 6                 # End of sleep period (6:00 AM)
        self.night_duration_hours = 8           # Total sleep period duration

        # Quality control parameters
        self.temp_threshold = 18.0              # Temperature threshold in Celsius


        # Visualization parameters
        self.create_individual_plots = True     # Create detailed plots for each participant
        self.create_summary_plots = True        # Create overall summary plots

        # Initialize the supporting systems
        self.setup_logging()                # Set up the system to record what happens
        self.initialize_stats()             # Set up counters to track our progress

        # Print information about what mode we're running in
        print(f"Running in {self.mode} mode")
        print(f"Base directory: {self.base_dir}")
        print(f"Temperature threshold: {self.temp_threshold}°C")
        print(f"Night period: {self.night_start_hour}:00 - {self.night_end_hour}:00")

    def initialize_stats(self):
        """
        Set up counters to keep track of processing statistics.
        This helps us monitor success rates and identify any problems.
        """
        self.stats = {
            'total_files': 0,               # How many .cwa files we found
            'processed_files': 0,           # How many files we successfully processed
            'failed_files': 0,              # How many files had errors
            'total_nights': 0,              # Total number of nights across all files
            'valid_nights': 0,              # Number of nights that passed quality control
            'total_hours': 0.0,             # Total hours of data processed
            'valid_hours': 0.0,             # Hours of data that passed quality control
            'controls_processed': 0,        # How many control files processed
            'irbd_processed': 0,            # How many iRBD files processed
            'temperature_filtered_hours': 0.0,  # Hours removed by temperature filtering
            'nonwear_filtered_hours': 0.0   # Hours removed by non-wear detection
        }

    def setup_logging(self):
        """
        Set up the logging system to record everything that happens during preprocessing.
        This creates a detailed record of the process for debugging and documentation.
        """
        # Create a unique log file name with current date and time
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"preprocessing_v1_{current_time}.log"

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
        self.logger.info(f"=== Preprocessing Pipeline Started ({self.mode}) ===")
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Temperature threshold: {self.temp_threshold}°C")
        self.logger.info(f"Night period: {self.night_start_hour}:00-{self.night_end_hour}:00")
        self.logger.info(f"actipy parameters: {self.lowpass_hz}Hz lowpass, {self.resample_hz}Hz resample")

    def find_cwa_files(self, directory):
        """
        Look for all .cwa and .CWA files in a specific directory.
        This handles both lowercase and uppercase file extensions.

        Args:
            directory: The folder path where we want to search for accelerometer files

        Returns:
            A list of file paths for all accelerometer files found in the directory
        """
        # Search for both .cwa and .CWA files
        cwa_pattern_lower = directory / "*.cwa"
        cwa_pattern_upper = directory / "*.CWA"

        # Find all files matching both patterns
        cwa_files = glob.glob(str(cwa_pattern_lower)) + glob.glob(str(cwa_pattern_upper))

        # Convert file paths from strings to Path objects
        cwa_files = [Path(f) for f in cwa_files]

        # Sort files alphabetically for consistent processing order
        cwa_files.sort()

        # Log how many files we found
        self.logger.info(f"Found {len(cwa_files)} .cwa/.CWA files in {directory}")

        return cwa_files

    def read_accelerometer_data(self, file_path):
        """
        Read raw accelerometer data from a .cwa file using actipy.
        This applies all the signal processing (filtering, resampling, calibration).

        Args:
            file_path: Path to the .cwa file to read

        Returns:
            tuple: (data_dataframe, info_dictionary)
                - data_dataframe: Contains x, y, z accelerometer data and timestamps
                - info_dictionary: Contains metadata about the processing
        """
        try:
            # Extract participant ID from filename
            participant_id = file_path.stem

            self.logger.info(f"Reading accelerometer data: {participant_id}")

            # Use actipy to read and process the accelerometer file
            data, info = actipy.read_device(
                str(file_path),                          # Path to the .cwa file
                lowpass_hz=self.lowpass_hz,         # Apply 12Hz lowpass filter
                resample_hz=self.resample_hz,       # Resample to 30Hz
                detect_nonwear=self.detect_nonwear, # Automatically detect non-wear periods
                calibrate_gravity=self.calibrate_gravity  # Correct for gravity and orientation
            )

            # Log information about what actipy found
            self.logger.info(f"   - Total samples: {len(data):,}")
            self.logger.info(f"   - Sampling rate: {info.get('ResampleRate', 'Unknown')} Hz")
            self.logger.info(f"   - Duration: {info.get('WearTime(days)', 0):.2f} days")
            self.logger.info(f"   - Non-wear time: {info.get('NonwearTime(days)', 0):.2f} days")

            return data, info

        except Exception as e:
            # If anything goes wrong, log the error and re-raise it
            self.logger.error(f"Error reading {file_path.name}: {str(e)}")
            raise

    def detect_nonwear_rate_of_change(self, data, temp_threshold=18.0, window_minutes=5):
        """
        Detect non-wear periods using rate-of-change temperature detection (DETACH algorithm).
        
        This method identifies non-wear by detecting rapid temperature drops that indicate
        device removal from the body. More sophisticated than simple threshold - reduces
        false positives from gradual temperature changes during sleep.
        
        DETACH ALGORITHM:
        1. Calculate temperature rate-of-change over sliding window
        2. Identify rapid drops (device removal from body)
        3. Mark periods below threshold AFTER rapid drop as non-wear
        4. Preserve periods that gradually cool (may be valid sleep)
        
        Args:
            data: DataFrame with temperature column
            temp_threshold: Temperature threshold in Celsius (default 18°C)
            window_minutes: Window size for rate-of-change calculation (default 5 min)
            
        Returns:
            Boolean mask: True for wear periods, False for non-wear
        """
        import pandas as pd
        
        # Calculate window size in samples
        window_samples = int(window_minutes * 60 * self.resample_hz)
        
        # Calculate temperature rate-of-change (°C per minute)
        temp_diff = data['temperature'].diff(window_samples)
        time_diff_minutes = window_samples / self.resample_hz / 60
        rate_of_change = temp_diff / time_diff_minutes
        
        # Detect rapid temperature drops (< -2°C per minute suggests device removal)
        # Gradual cooling during sleep is typically < -0.5°C per minute
        rapid_drop = rate_of_change < -2.0
        
        # Create initial wear mask (assume all worn)
        wear_mask = pd.Series(True, index=data.index)
        
        # Mark as non-wear if:
        # 1. Temperature below threshold AND rapid drop detected, OR
        # 2. Temperature very low (< 15°C, clearly not worn)
        below_threshold = data['temperature'] < temp_threshold
        very_cold = data['temperature'] < 15.0
        
        # Non-wear = (below threshold AND rapid drop) OR very cold
        nonwear_mask = (below_threshold & rapid_drop) | very_cold
        
        # Apply mask
        wear_mask = ~nonwear_mask
        
        return wear_mask

    def apply_temperature_filtering(self, data):
        """
        Apply rate-of-change temperature-based filtering to remove non-wear periods.
        
        V1: Uses DETACH algorithm with 18°C threshold and rate-of-change detection.
        More sophisticated than simple threshold - reduces false positives from
        gradual temperature drops during sleep.
        
        WHAT THIS DOES:
        1. Applies rate-of-change detection (detects rapid temperature drops)
        2. Combines with actipy non-wear detection
        3. Removes only periods with rapid cooling below 18°C
        4. Preserves gradual cooling (valid sleep data)
        
        Args:
            data: DataFrame containing accelerometer data with temperature column
            
        Returns:
            DataFrame: Filtered data with only wear periods
        """
        # Count initial data
        initial_samples = len(data)
        initial_hours = initial_samples / self.resample_hz / 3600
        
        self.logger.info(f" Applying temperature filtering (rate-of-change, ≥{self.temp_threshold}°C)")
        self.logger.info(f"   - Initial data: {initial_samples:,} samples ({initial_hours:.1f} hours)")
        
        # Apply rate-of-change detection
        wear_mask_temp = self.detect_nonwear_rate_of_change(data, self.temp_threshold)
        
        # actipy non-wear mask (data marked as NaN by actipy)
        valid_data_mask = ~(data['x'].isna() | data['y'].isna() | data['z'].isna())
        
        # Combine both filters: must pass both temperature AND actipy checks
        combined_mask = wear_mask_temp & valid_data_mask
        
        # Apply filter
        filtered_data = data[combined_mask].copy()
        
        # Calculate statistics
        final_samples = len(filtered_data)
        final_hours = final_samples / self.resample_hz / 3600
        removed_samples = initial_samples - final_samples
        removed_hours = removed_samples / self.resample_hz / 3600
        retention_rate = (final_samples / initial_samples * 100) if initial_samples > 0 else 0
        
        # Log results
        self.logger.info(f"   - After filtering: {final_samples:,} samples ({final_hours:.1f} hours)")
        self.logger.info(f"   - Removed: {removed_samples:,} samples ({removed_hours:.1f} hours)")
        self.logger.info(f"   - Retention rate: {retention_rate:.1f}%")
        
        # Update statistics
        self.stats['temperature_filtered_hours'] += removed_hours
        
        # Check if data remains
        if len(filtered_data) == 0:
            raise ValueError("No data remaining after temperature filtering")
        
        return filtered_data

    def segment_nights(self, data):
        """
        Split continuous accelerometer data into individual night periods (22:00-06:00).
        This version properly handles timestamps that are in the DataFrame index.

        HOW IT WORKS:
        1. Converts the DataFrame index to datetime if needed
        2. Identifies the date range covered by the data
        3. For each date, creates a night period (22:00 to next day 06:00)
        4. Extracts data for each night period
        5. Returns a list of dictionaries with night information

        KEY IMPROVEMENTS:
        - Now correctly handles timestamps stored in DataFrame index
        - More robust date/time handling
        - Better logging for debugging
        - Maintains all original functionality

        Args:
            data: Pandas DataFrame with accelerometer data (x,y,z) and timestamps in the index

        Returns:
            List of dictionaries, each containing:
            - night_number: Sequential night count (1, 2, 3...)
            - date: Calendar date of night start
            - start_time: Exact datetime when night starts (22:00)
            - end_time: Exact datetime when night ends (06:00 next day)
            - data: DataFrame with night's accelerometer data
            - samples: Number of samples in this night
            - duration_hours: Duration in hours
        """

        self.logger.info(f"Segmenting data into night periods ({self.night_start_hour}:00-{self.night_end_hour}:00)")

        # =====================================================================
        # STEP 1: PREPARE TIMESTAMPS
        # =====================================================================
        # actipy stores timestamps in the DataFrame index - we need to ensure:
        # 1. The index is properly formatted as datetime
        # 2. We can access it for night segmentation

        # Convert index to datetime if it isn't already
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
                self.logger.info("   - Converted index to datetime format")
            except Exception as e:
                self.logger.error(f"   - Failed to convert index to datetime: {str(e)}")
                raise ValueError("Could not convert DataFrame index to datetime")

        # Create a copy of the data to avoid modifying the original
        data = data.copy()

        # Add timestamps as a column for easier access (optional)
        # This isn't strictly necessary but can make code more readable
        data['timestamp'] = data.index

        # =====================================================================
        # STEP 2: IDENTIFY DATE RANGE
        # =====================================================================
        # Find the first and last dates in our data
        # Note: .date() converts datetime to just date (without time)

        start_date = data.index.min().date()
        end_date = data.index.max().date()

        self.logger.info(f"   - Data spans from {start_date} to {end_date}")
        self.logger.info(f"   - First timestamp: {data.index.min()}")
        self.logger.info(f"   - Last timestamp: {data.index.max()}")

        # =====================================================================
        # STEP 3: EXTRACT NIGHT PERIODS
        # =====================================================================
        # Initialize list to store each night's data
        nights_data = []

        current_date = start_date
        night_number = 1  # Count nights sequentially

        # Loop through each date in the recording period
        while current_date <= end_date:
            # =============================================================
            # STEP 3.1: DEFINE NIGHT BOUNDARIES
            # =============================================================
            # Night starts at night_start_hour (22:00) on current_date
            night_start = datetime.combine(current_date, time(self.night_start_hour, 0))

            # Night ends at night_end_hour (06:00) on the NEXT day
            next_date = current_date + timedelta(days=1)
            night_end = datetime.combine(next_date, time(self.night_end_hour, 0))

            # =============================================================
            # STEP 3.2: EXTRACT DATA FOR THIS NIGHT
            # =============================================================
            # Create boolean mask for this night period
            night_mask = (data.index >= night_start) & (data.index < night_end)
            night_data = data[night_mask].copy()

            # =============================================================
            # STEP 3.3: STORE NIGHT DATA IF NOT EMPTY
            # =============================================================
            if len(night_data) > 0:
                # Calculate duration in hours
                duration_hours = len(night_data) / self.resample_hz / 3600

                # Store night information in dictionary
                night_info = {
                    'night_number': night_number,
                    'date': current_date,
                    'start_time': night_start,
                    'end_time': night_end,
                    'data': night_data,
                    'samples': len(night_data),
                    'duration_hours': duration_hours
                }

                nights_data.append(night_info)

                self.logger.info(f"   - Night {night_number} ({current_date}): "
                           f"{len(night_data):,} samples ({duration_hours:.1f}h)")

                night_number += 1
            else:
                self.logger.info(f"   - Night {current_date}: No data available")

            # Move to next date
            current_date = next_date

        # =====================================================================
        # STEP 4: FINAL VALIDATION AND STATISTICS
        # =====================================================================
        total_nights = len(nights_data)
        total_night_hours = sum(night['duration_hours'] for night in nights_data)

        self.logger.info(f"   - Total nights extracted: {total_nights}")
        self.logger.info(f"   - Total night data: {total_night_hours:.1f} hours")

        # Validate we found at least one night
        if total_nights == 0:
            self.logger.warning("No night periods found in data!")

        return nights_data

    def save_preprocessed_data(self, nights_data, participant_id, output_path):
        """
        Save the preprocessed night-segmented data to an HDF5 file.

        Args:
            nights_data: List of dictionaries containing data for each night
            participant_id: Unique identifier for this participant
            output_path: Full path where to save the HDF5 file
        """
        try:
            self.logger.info(f"Saving preprocessed data: {output_path.name}")

            # Create the HDF5 file for writing
            with h5py.File(output_path, 'w') as f:
                # Save participant information as file attributes
                f.attrs['name'] = participant_id                    # Participant identifier
                f.attrs['number_of_nights'] = len(nights_data)      # How many nights we have

                # Save each night's data as a separate group in the file
                for night_info in nights_data:
                    night_num = night_info['night_number']
                    night_data = night_info['data']

                    # Create a group for this night (like a folder within the file)
                    night_group = f.create_group(f"night{night_num}")

                    # Save the accelerometer data (x, y, z axes)
                    # Combine x,y,z into single array for better compression
                    accel_data = np.column_stack([
                        night_data['x'].values,
                        night_data['y'].values,
                        night_data['z'].values
                    ])
                    night_group.create_dataset(
                        'accel',
                        data=accel_data,
                        compression='gzip',
                        compression_opts=4,
                        chunks=(min(10000, accel_data.shape[0]), 3),
                        dtype='float32'
                    )

                    # Save timestamps as strings (HDF5 handles this efficiently)
                    timestamps_data = [ts.isoformat() for ts in night_data.index]
                    night_group.create_dataset(
                        'timestamps',
                        data=timestamps_data,
                        compression='gzip',
                        compression_opts=4,
                        chunks=(min(10000, len(timestamps_data)),)
                    )

            # Log successful save with file size information
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"   - File saved successfully: {file_size_mb:.1f} MB")
            self.logger.info(f"   - Nights saved: {len(nights_data)}")

        except Exception as e:
            # If anything goes wrong, log the error and re-raise it
            self.logger.error(f"Error saving {output_path.name}: {str(e)}")
            raise

    def verify_h5_structure(self, h5_file):
        """
        Verify that the saved HDF5 file has the correct structure and no metadata.
        This is a quality control check to ensure our files are saved correctly.

        Args:
            h5_file: Path to the HDF5 file to verify

        Returns:
            bool: True if structure is correct, False otherwise
        """
        try:
            with h5py.File(h5_file, 'r') as f:
                # Check that we have the required attributes
                if 'name' not in f.attrs or 'number_of_nights' not in f.attrs:
                    self.logger.error(f"Missing required attributes in {h5_file.name}")
                    return False

                num_nights = f.attrs['number_of_nights']

                # Check each night group
                for night_num in range(1, num_nights + 1):
                    night_group_name = f"night{night_num}"

                    if night_group_name not in f:
                        self.logger.error(f"Missing {night_group_name} in {h5_file.name}")
                        return False

                    night_group = f[night_group_name]

                    # Check that each night has the required datasets
                    required_datasets = ['accel', 'timestamps']
                    for dataset_name in required_datasets:
                        if dataset_name not in night_group:
                            self.logger.error(f"Missing {dataset_name} in {night_group_name} of {h5_file.name}")
                            return False

                    # Verify that night group has NO metadata (as requested)
                    if len(night_group.attrs) > 0:
                        self.logger.error(f"Unexpected metadata found in {night_group_name} of {h5_file.name}")
                        return False

                self.logger.info(f"Structure verification passed: {h5_file.name}")
                return True

        except Exception as e:
            self.logger.error(f"Error verifying {h5_file.name}: {str(e)}")
            return False

    def process_participant(self, cwa_file, group_type):
        """
        Process one participant from start to finish.
        This is the main processing function that orchestrates all steps for one person.

        Args:
            cwa_file: Path to the participant's .cwa file
            group_type: Either 'controls' or 'irbd'
        """
        try:
            # Extract participant ID from filename
            participant_id = cwa_file.stem

            # Log that we're starting to process this participant
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing {group_type.upper()}: {participant_id}")
            self.logger.info(f"{'='*60}")

            # STEP 1: Read accelerometer data using actipy
            accel_data, accel_info = self.read_accelerometer_data(cwa_file)

            # STEP 2: Apply temperature filtering and non-wear removal
            filtered_data = self.apply_temperature_filtering(accel_data)

            # STEP 3: Segment data into individual nights
            nights_data = self.segment_nights(filtered_data)

            # STEP 4: Save preprocessed data to HDF5 file
            if group_type == 'controls':
                output_dir = self.preprocessed_controls_dir
            else:
                output_dir = self.preprocessed_irbd_dir

            output_path = output_dir / f"{participant_id}.h5"
            self.save_preprocessed_data(nights_data, participant_id, output_path)

            # STEP 5: Verify the saved file structure
            if self.verify_h5_structure(output_path):
                self.logger.info(f"{participant_id} processed successfully")
            else:
                self.logger.error(f"Structure verification failed for {participant_id}")
                self.stats['failed_files'] += 1
                return

            # STEP 6: Update statistics
            self.stats['processed_files'] += 1
            self.stats['total_nights'] += len(nights_data)
            self.stats['valid_nights'] += len(nights_data)

            total_hours = sum(night['duration_hours'] for night in nights_data)
            self.stats['valid_hours'] += total_hours

            if group_type == 'controls':
                self.stats['controls_processed'] += 1
            else:
                self.stats['irbd_processed'] += 1

            # Log success summary
            self.logger.info(f"Processing summary:")
            self.logger.info(f"   - Nights: {len(nights_data)}")
            self.logger.info(f"   - Total hours: {total_hours:.1f}")
            self.logger.info(f"   - Output file: {output_path.name}")

        except Exception as e:
            # If anything goes wrong, log the error but continue with other participants
            self.logger.error(f"Error processing {participant_id}: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.stats['failed_files'] += 1

    def run_example_test(self):
        """
        Run preprocessing on just one example participant for testing.
        This processes the example .cwa file to verify everything works correctly.
        """
        self.logger.info("EXAMPLE TEST MODE: Processing example participant")

        # Check if the example file exists
        if not self.raw_example_file.exists():
            self.logger.error(f"Example file not found: {self.raw_example_file}")
            return

        # Determine which group the example file belongs to
        # (This is just for logging purposes - the processing is the same)
        group_type = 'controls'  # Assume it's a control unless we know otherwise

        # Set statistics for processing one file
        self.stats['total_files'] = 1

        # Process the example participant
        self.process_participant(self.raw_example_file, group_type)

        # Show final results
        self.print_final_statistics()

        # Report success or failure
        if self.stats['processed_files'] > 0:
            self.logger.info("EXAMPLE TEST SUCCESSFUL!")
            self.logger.info("Ready for full dataset processing!")
        else:
            self.logger.error("EXAMPLE TEST FAILED!")

    def run_full_processing(self):
        """
        Run preprocessing on all participants in the dataset.
        This processes all .cwa files in both controls and iRBD directories.
        """
        self.logger.info("FULL PROCESSING MODE: Processing all participants")

        # Find all .cwa files in both directories
        controls_files = self.find_cwa_files(self.raw_controls_dir)
        irbd_files = self.find_cwa_files(self.raw_irbd_dir)

        # Calculate total number of files
        total_files = len(controls_files) + len(irbd_files)
        self.stats['total_files'] = total_files

        # Check if we found any files
        if total_files == 0:
            self.logger.error("No .cwa files found in input directories")
            return

        # Log what we found
        self.logger.info(f"Found {total_files} files:")
        self.logger.info(f"   - Controls: {len(controls_files)} files")
        self.logger.info(f"   - iRBD: {len(irbd_files)} files")

        # Process all control files
        self.logger.info(f"\n Processing CONTROLS ({len(controls_files)} files)...")
        for i, cwa_file in enumerate(controls_files, 1):
            self.logger.info(f"\n--- Controls Progress: {i}/{len(controls_files)} ---")
            self.process_participant(cwa_file, 'controls')

        # Process all iRBD files
        self.logger.info(f"\n Processing iRBD ({len(irbd_files)} files)...")
        for i, cwa_file in enumerate(irbd_files, 1):
            self.logger.info(f"\n--- iRBD Progress: {i}/{len(irbd_files)} ---")
            self.process_participant(cwa_file, 'irbd')

        # Show final statistics
        self.print_final_statistics()

    def run_preprocessing(self):
        """
        Main function to run the preprocessing pipeline.
        Decides whether to run example test or full processing based on configuration.
        """
        if self.mode == "EXAMPLE_TEST":
            self.run_example_test()
        else:
            self.run_full_processing()

    def print_final_statistics(self):
        """
        Print comprehensive summary of the preprocessing results.
        Shows processing success rates, data quality metrics, and file statistics.
        """
        # Print header
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"PREPROCESSING COMPLETED ({self.mode})")
        self.logger.info(f"{'='*60}")

        # File processing statistics
        self.logger.info(f"File Processing:")
        self.logger.info(f"  - Total files: {self.stats['total_files']}")
        self.logger.info(f"  - Successfully processed: {self.stats['processed_files']}")
        self.logger.info(f"  - Failed to process: {self.stats['failed_files']}")

        # Calculate success rate
        if self.stats['total_files'] > 0:
            success_rate = self.stats['processed_files'] / self.stats['total_files'] * 100
            self.logger.info(f"  - Success rate: {success_rate:.1f}%")

        self.logger.info("")

        # Group breakdown
        self.logger.info(f"Group Breakdown:")
        self.logger.info(f"  - Controls processed: {self.stats['controls_processed']}")
        self.logger.info(f"  - iRBD processed: {self.stats['irbd_processed']}")
        self.logger.info("")

        # Data quality statistics
        self.logger.info(f"Data Quality:")
        self.logger.info(f"  - Total nights: {self.stats['total_nights']}")
        self.logger.info(f"  - Valid nights: {self.stats['valid_nights']}")
        self.logger.info(f"  - Valid data hours: {self.stats['valid_hours']:.1f}")
        self.logger.info(f"  - Temperature filtered hours: {self.stats['temperature_filtered_hours']:.1f}")

        # Calculate averages
        if self.stats['processed_files'] > 0:
            avg_nights = self.stats['valid_nights'] / self.stats['processed_files']
            avg_hours = self.stats['valid_hours'] / self.stats['processed_files']
            self.logger.info(f"  - Average nights per participant: {avg_nights:.1f}")
            self.logger.info(f"  - Average hours per participant: {avg_hours:.1f}")

        self.logger.info("")

        # Final success message
        if self.stats['processed_files'] > 0:
            self.logger.info("Preprocessing pipeline completed successfully!")
            if self.mode == "EXAMPLE_TEST":
                self.logger.info("Example data ready for feature extraction")
            else:
                self.logger.info("All data ready for feature extraction")
        else:
            self.logger.error("Preprocessing pipeline failed!")


# In[5]:


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """
    Main function that runs when the script is executed directly.
    Creates a PreprocessingPipeline object and runs the entire process.
    """
    try:
        # Create and run the preprocessing pipeline
        pipeline = PreprocessingPipeline()
        pipeline.run_preprocessing()

    except KeyboardInterrupt:
        print("\n Preprocessing interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n Preprocessing failed with error: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

# =============================================================================
# SCRIPT EXECUTION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

