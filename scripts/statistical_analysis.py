#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# FEATURES STATISTICAL ANALYSIS
# Statistical analysis of SSL-Wearables features between controls and iRBD groups.
# Statistically check if the average features across the night is different between iRBD and controls.

## INPUT
# Source : SSL-Wearables feature vectors from feature extraction pipeline
# Directories : 
#    - /work3/s184484/iRBD-detection/data/features/controls/
#    - /work3/s184484/iRBD-detection/data/features/irbd/
# Format : .npy files with night-structured features (nights × windows_per_night × 1024)


## PIPELINE
# 1. Feature Loading and Aggregation :
#    - Load individual participant feature files from both groups
#    - Calculate participant-level feature averages (average across all nights and windows)
#    - Create two groups: controls vs iRBD participants
#    - Ensure proper data structure for statistical testing
# 2. Statistical Testing Framework :
#    - Test normality for each of the 1024 features using Shapiro-Wilk test
#    - Apply appropriate statistical tests based on normality:
#         - Independent t-test for normally distributed features
#         - Mann-Whitney U test for non-normally distributed features
#    - Calculate effect sizes (Cohen's d for t-tests, rank-biserial correlation for Mann-Whitney)
#    - Apply Bonferroni correction for multiple comparisons (α = 0.05/1024)
# 3. Effect Size Analysis :
#    - Identify features with large effect sizes (Cohen's d ≥ 0.5)
#    - Rank features by effect size to find most discriminative patterns
#    - Combine statistical significance with practical significance
# 4. Results Interpretation :
#    - Identify features that are both statistically significant AND have large effect sizes
#    - Provide clinical interpretation of significant movement pattern differences
#    - Generate comprehensive summary statistics and visualizations


## OUTPUT
# Format : Statistical results, effect sizes, and comprehensive visualizations
# Directories :
#    - /work3/s184484/iRBD-detection/results/statistical_analysis/
#    - /work3/s184484/iRBD-detection/results/visualizations/


## ENVIRONMENT : env_insights


## HPC JOB EXECUTION
# Job script : statistical_analysis_job.sh
# Location : /work3/s184484/iRBD-detection/jobs/scripts/statistical_analysis_job.sh
# Queue : hpc (CPU nodes, high memory for 1024-dimensional analysis)
# Resources : 8 cores, 4GB RAM per core (32GB total for large feature matrices)
# Time limit : 6 hours
# Output logs : /work3/s184484/iRBD-detection/jobs/logs/stats/stats_output_JOBID.out
# Error logs : /work3/s184484/iRBD-detection/jobs/logs/stats/stats_error_JOBID.err


# In[ ]:


# Basic Python libraries for file operations and system control
import os                    # Operating System interface - helps us work with files and folders
import sys                   # System-specific parameters - helps us control the program execution
import numpy as np           # NumPy - for mathematical operations on arrays of numbers
import pandas as pd          # Pandas - for working with data tables and organizing information
from datetime import datetime, timedelta  # For working with dates and times
import logging               # For creating detailed log files that record what the program does
from pathlib import Path     # For easier and more reliable file path handling
import glob                  # For finding files that match specific patterns
import traceback             # For showing detailed error messages when something goes wrong
import json                  # For saving and loading JSON files (configuration and metadata)
import pickle                # For saving Python objects (like statistical results)
import gc                    # Garbage collection - for managing memory usage
import warnings              # For controlling warning messages
warnings.filterwarnings('ignore')  # Suppress unnecessary warnings

# Statistical analysis libraries
import scipy.stats as stats  # For statistical tests (t-tests, Mann-Whitney U, etc.)
from scipy.stats import (    # Specific statistical functions we'll use
    shapiro,                 # Shapiro-Wilk test for checking if data is normally distributed
    mannwhitneyu,           # Mann-Whitney U test (non-parametric alternative to t-test)
    ttest_ind,              # Independent t-test for comparing two groups
    pearsonr,               # Pearson correlation coefficient
    spearmanr,              # Spearman correlation (non-parametric)
    normaltest,             # D'Agostino's normality test (alternative to Shapiro-Wilk)
    levene                  # Levene's test for equal variances
)
from statsmodels.stats.multitest import multipletests  # Multiple comparison correction methods
import statsmodels.api as sm  # Statistical models for advanced analysis

# WHAT IS STATISTICAL TESTING:
# Statistical testing helps us determine if differences between groups are real
# or just due to random chance. We use p-values to quantify this uncertainty.

# WHAT IS EFFECT SIZE:
# Effect size tells us how big the difference is between groups, regardless of
# statistical significance. Large effect sizes indicate practically meaningful differences.

# WHAT IS MULTIPLE COMPARISON CORRECTION:
# When testing many features (1024), we need to adjust our significance threshold
# to avoid finding false positives due to chance alone.

# Machine learning libraries for data preprocessing
from sklearn.preprocessing import StandardScaler  # For normalizing data (z-score standardization)
from sklearn.decomposition import PCA            # Principal Component Analysis for dimensionality reduction
from sklearn.manifold import TSNE                # t-SNE for visualization of high-dimensional data

# Visualization libraries for creating plots and charts
import matplotlib.pyplot as plt    # Main plotting library - like creating graphs in Excel
import seaborn as sns             # Statistical plotting library - makes beautiful, professional plots

# Configure matplotlib and seaborn for professional-looking plots
plt.style.use('seaborn-v0_8')     # Use seaborn's visual style (makes plots look professional)
sns.set_palette("husl")           # Set a nice color palette (colors that work well together)
plt.rcParams['figure.figsize'] = (12, 8)  # Set default size for all plots (12 inches wide, 8 inches tall)
plt.rcParams['font.size'] = 10    # Set default font size for all text in plots
plt.rcParams['axes.grid'] = True  # Show grid lines on plots for easier reading


# In[ ]:


# =============================================================================
# STATISTICAL ANALYSIS PIPELINE CLASS
# =============================================================================

class FeatureStatisticalAnalysis:
    """
    Comprehensive statistical analysis pipeline for comparing SSL-Wearables features 
    between iRBD patients and healthy controls.

    WHAT THIS CLASS DOES:
    This class performs a thorough statistical comparison of movement patterns between
    iRBD patients and healthy controls using the 1024-dimensional SSL-Wearables features.
    It identifies which specific movement characteristics are significantly different
    between the two groups and quantifies the size of these differences.

    WHY THIS ANALYSIS IS IMPORTANT:
    - Identifies specific movement patterns that distinguish iRBD from normal sleep
    - Provides scientific evidence for the effectiveness of the detection approach
    - Helps understand the underlying pathophysiology of iRBD
    - Validates that the SSL-Wearables features capture clinically relevant information
    - Supports the development of better diagnostic tools

    STATISTICAL APPROACH:
    1. Calculate participant-level averages (reduces night-to-night variability)
    2. Test each feature for normality to choose appropriate statistical tests
    3. Compare groups using t-tests or Mann-Whitney U tests as appropriate
    4. Calculate effect sizes to quantify practical significance
    5. Apply multiple comparison correction to control false discovery rate
    6. Identify features with both statistical and practical significance

    NOTE: This analysis requires multiple participants per group (controls and iRBD).
    It cannot be tested with single example files like preprocessing or feature extraction.
    """

    def __init__(self):
        """
        Initialize the statistical analysis pipeline with all necessary configuration.
        This sets up directories, parameters, and statistical frameworks.
        """

        # =================================================================
        # DIRECTORY SETUP - Define where to find files and save results
        # =================================================================

        # Main project directory on HPC
        self.base_dir = Path("/work3/s184484/iRBD-detection")

        # Input directories (where the features are stored):
        self.features_dir = self.base_dir / "data" / "features"                    # Main features directory
        self.features_controls_dir = self.features_dir / "controls"               # Individual control features
        self.features_irbd_dir = self.features_dir / "irbd"                       # Individual iRBD features

        # Output directories (where to save statistical analysis results):
        self.results_dir = self.base_dir / "results"                              # Main results directory
        self.stats_results_dir = self.results_dir / "statistical_analysis"       # Statistical analysis results

        # Visualization output directory (where to save plots for the report):
        self.plots_dir = self.results_dir / "visualizations"

        # Only create log directory if it doesn't exist (for logging only)
        self.log_dir = self.base_dir / "validation" / "data_quality_reports"
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # =================================================================
        # STATISTICAL PARAMETERS - Settings that control the analysis
        # =================================================================

        # Multiple comparison correction parameters:
        self.alpha = 0.05                           # Overall significance level (5%)
        self.feature_dim = 1024                     # Number of SSL-Wearables features to test
        self.bonferroni_alpha = self.alpha / self.feature_dim  # Bonferroni corrected alpha

        # WHY BONFERRONI CORRECTION:
        # When testing 1024 features simultaneously, we have a high chance of finding
        # false positives by random chance alone. Bonferroni correction divides our
        # significance threshold by the number of tests to control this.
        # New threshold: 0.05/1024 = 4.88e-05 (much more stringent)

        # Effect size thresholds (Cohen's conventions for interpreting effect sizes):
        self.small_effect = 0.2                     # Small effect size (subtle difference)
        self.medium_effect = 0.5                    # Medium effect size (moderate difference)
        self.large_effect = 0.8                     # Large effect size (substantial difference)

        # WHY COHEN'S d ≥ 0.5 AS THRESHOLD:
        # Medium effect sizes (≥0.5) are generally considered clinically meaningful.
        # This ensures we focus on features with practical, not just statistical, significance.
        # Small effects might be statistically significant but not clinically useful.

        # Normality testing parameters:
        self.normality_alpha = 0.05                 # Significance level for normality tests
        self.min_sample_size = 5                    # Minimum sample size for statistical tests

        # WHY TEST FOR NORMALITY:
        # Different statistical tests are appropriate for normal vs non-normal data:
        # - Normal data: t-test (more powerful, assumes normal distribution)
        # - Non-normal data: Mann-Whitney U (robust, no distribution assumptions)

        # Data aggregation parameters:
        self.aggregation_method = 'mean'            # How to aggregate features across nights ('mean' or 'median')

        # WHY USE MEAN AGGREGATION:
        # Taking the mean across all nights for each participant gives us a stable
        # estimate of their typical movement patterns, reducing night-to-night variability
        # while preserving individual differences.

        # Initialize the supporting systems
        self.setup_logging()                # Set up the system to record what happens
        self.initialize_stats()             # Set up counters to track our progress

        # Print information about the analysis configuration
        print(f"Statistical Analysis Pipeline Initialized")
        print(f"Base directory: {self.base_dir}")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Bonferroni corrected alpha: {self.bonferroni_alpha:.2e}")
        print(f"Effect size threshold: {self.medium_effect} (medium)")
        print(f"Aggregation method: {self.aggregation_method}")

    def initialize_stats(self):
        """
        Set up counters to keep track of analysis statistics.
        This helps us monitor progress and results throughout the analysis.
        """
        self.stats = {
            # Dataset information
            'total_participants': 0,        # How many participants we analyzed
            'controls_count': 0,            # Number of control participants
            'irbd_count': 0,                # Number of iRBD participants
            'total_nights': 0,              # Total nights across all participants
            'total_windows': 0,             # Total 10-second windows processed

            # Feature analysis results
            'total_features_tested': 0,     # Number of features tested (should be 1024)
            'normal_features': 0,           # Number of features with normal distribution
            'non_normal_features': 0,       # Number of features with non-normal distribution
            'significant_features': 0,      # Number of statistically significant features
            'large_effect_features': 0,     # Number of features with large effect sizes
            'both_sig_and_large': 0,        # Features that are both significant AND have large effects

            # Analysis performance
            'analysis_time': 0.0,           # Total analysis time in seconds
            'memory_usage': 0.0,            # Peak memory usage during analysis

            # Top results
            'top_features': [],             # Most important features identified
            'effect_size_distribution': [], # Distribution of effect sizes across features
            'p_value_distribution': []      # Distribution of p-values across features
        }

    def setup_logging(self):
        """
        Set up the logging system to record everything that happens during analysis.
        This creates a detailed record of the process for debugging and documentation.
        """
        # Create a unique log file name with current date and time
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"statistical_analysis_{current_time}.log"

        # Configure the logging system to write to both file and console
        logging.basicConfig(
            level=logging.INFO,                     # Log all INFO level messages and above
            format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp and level
            handlers=[
                logging.FileHandler(log_file),      # Save messages to log file
                logging.StreamHandler(sys.stdout)   # Also display on screen
            ]
        )

        # Create our logger object
        self.logger = logging.getLogger(__name__)

        # Write initial log messages to document the start of analysis
        self.logger.info(f"=== Statistical Analysis Pipeline Started ===")
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Feature dimension: {self.feature_dim}")
        self.logger.info(f"Bonferroni corrected alpha: {self.bonferroni_alpha:.2e}")
        self.logger.info(f"Effect size threshold: {self.medium_effect}")
        self.logger.info(f"Aggregation method: {self.aggregation_method}")

    def load_and_aggregate_features(self):
        """
        Load feature data for all participants and calculate participant-level averages.

        WHAT THIS FUNCTION DOES:
        1. Loads individual participant feature files from both groups
        2. For each participant, calculates the average across all nights and windows
        3. Creates two arrays: one for controls, one for iRBD patients
        4. Each participant contributes one 1024-dimensional feature vector

        WHY AGGREGATE ACROSS NIGHTS:
        - Reduces night-to-night variability within participants
        - Creates stable estimates of individual movement patterns
        - Enables participant-level statistical comparisons
        - Simplifies the statistical analysis (one value per participant per feature)

        Returns:
            tuple: (controls_features, irbd_features, participant_info)
                - controls_features: Array of shape (n_controls, 1024)
                - irbd_features: Array of shape (n_irbd, 1024)
                - participant_info: Dictionary with participant details
        """
        try:
            self.logger.info("📖 Loading and aggregating feature data...")

            controls_features = []
            irbd_features = []
            participant_info = {
                'controls': [],
                'irbd': [],
                'controls_nights': [],
                'irbd_nights': []
            }

            # Load control participants
            self.logger.info("   Processing control participant features...")
            control_files = list(self.features_controls_dir.glob("*_features.npy"))

            if len(control_files) == 0:
                raise FileNotFoundError(f"No control feature files found in {self.features_controls_dir}")

            for file_path in sorted(control_files):
                try:
                    # Load participant data
                    # Each file contains: {'features': array, 'windows_mask': array, 'participant_id': str}
                    data = np.load(file_path, allow_pickle=True).item()

                    # Extract the components
                    features = data['features']  # Shape: (nights, max_windows_per_night, 1024)
                    windows_mask = data['windows_mask']  # Shape: (nights, max_windows_per_night)
                    participant_id = data['participant_id']

                    # Calculate participant-level feature averages
                    # We need to aggregate across all valid windows for this participant
                    valid_features = []

                    # Process each night separately
                    for night_idx in range(features.shape[0]):
                        night_features = features[night_idx]  # Shape: (max_windows_per_night, 1024)
                        night_mask = windows_mask[night_idx]  # Shape: (max_windows_per_night,)

                        # Get only the valid windows for this night (where mask is True)
                        valid_night_features = night_features[night_mask]  # Shape: (valid_windows, 1024)

                        # Only include nights with at least some valid data
                        if len(valid_night_features) > 0:
                            valid_features.append(valid_night_features)

                    # If we have valid data for this participant
                    if valid_features:
                        # Concatenate all valid features across all nights
                        all_features = np.concatenate(valid_features, axis=0)  # Shape: (total_valid_windows, 1024)

                        # Calculate the mean across all windows (aggregation step)
                        if self.aggregation_method == 'mean':
                            participant_avg_features = np.mean(all_features, axis=0)  # Shape: (1024,)
                        elif self.aggregation_method == 'median':
                            participant_avg_features = np.median(all_features, axis=0)  # Shape: (1024,)

                        # Store the results
                        controls_features.append(participant_avg_features)
                        participant_info['controls'].append(participant_id)
                        participant_info['controls_nights'].append(len(valid_features))

                        self.logger.info(f"     - {participant_id}: {len(valid_features)} nights, "
                                       f"{len(all_features)} total windows")

                except Exception as e:
                    self.logger.error(f"     - Error loading {file_path.name}: {str(e)}")
                    continue

            # Load iRBD participants
            self.logger.info("   Processing iRBD participant features...")
            irbd_files = list(self.features_irbd_dir.glob("*_features.npy"))

            if len(irbd_files) == 0:
                raise FileNotFoundError(f"No iRBD feature files found in {self.features_irbd_dir}")

            for file_path in sorted(irbd_files):
                try:
                    # Load participant data (same structure as controls)
                    data = np.load(file_path, allow_pickle=True).item()

                    # Extract the components
                    features = data['features']  # Shape: (nights, max_windows_per_night, 1024)
                    windows_mask = data['windows_mask']  # Shape: (nights, max_windows_per_night)
                    participant_id = data['participant_id']

                    # Calculate participant-level feature averages (same process as controls)
                    valid_features = []

                    for night_idx in range(features.shape[0]):
                        night_features = features[night_idx]  # Shape: (max_windows_per_night, 1024)
                        night_mask = windows_mask[night_idx]  # Shape: (max_windows_per_night,)

                        # Get only the valid windows for this night
                        valid_night_features = night_features[night_mask]  # Shape: (valid_windows, 1024)

                        if len(valid_night_features) > 0:
                            valid_features.append(valid_night_features)

                    if valid_features:
                        # Concatenate and aggregate
                        all_features = np.concatenate(valid_features, axis=0)  # Shape: (total_valid_windows, 1024)

                        if self.aggregation_method == 'mean':
                            participant_avg_features = np.mean(all_features, axis=0)  # Shape: (1024,)
                        elif self.aggregation_method == 'median':
                            participant_avg_features = np.median(all_features, axis=0)  # Shape: (1024,)

                        # Store the results
                        irbd_features.append(participant_avg_features)
                        participant_info['irbd'].append(participant_id)
                        participant_info['irbd_nights'].append(len(valid_features))

                        self.logger.info(f"     - {participant_id}: {len(valid_features)} nights, "
                                       f"{len(all_features)} total windows")

                except Exception as e:
                    self.logger.error(f"     - Error loading {file_path.name}: {str(e)}")
                    continue

            # Convert lists to numpy arrays
            controls_features = np.array(controls_features)  # Shape: (n_controls, 1024)
            irbd_features = np.array(irbd_features)          # Shape: (n_irbd, 1024)

            # Check that we have sufficient data for analysis
            if len(controls_features) < self.min_sample_size:
                raise ValueError(f"Insufficient control participants: {len(controls_features)} < {self.min_sample_size}")

            if len(irbd_features) < self.min_sample_size:
                raise ValueError(f"Insufficient iRBD participants: {len(irbd_features)} < {self.min_sample_size}")

            # Update statistics
            self.stats['total_participants'] = len(controls_features) + len(irbd_features)
            self.stats['controls_count'] = len(controls_features)
            self.stats['irbd_count'] = len(irbd_features)
            self.stats['total_nights'] = (sum(participant_info['controls_nights']) + 
                                        sum(participant_info['irbd_nights']))

            # Log summary of loaded data
            self.logger.info(f"Feature data loaded and aggregated successfully:")
            self.logger.info(f"   - Controls: {len(controls_features)} participants")
            self.logger.info(f"   - iRBD: {len(irbd_features)} participants")
            self.logger.info(f"   - Total nights: {self.stats['total_nights']}")
            self.logger.info(f"   - Feature dimension: {controls_features.shape[1] if len(controls_features) > 0 else 0}")
            self.logger.info(f"   - Aggregation method: {self.aggregation_method}")

            return controls_features, irbd_features, participant_info

        except Exception as e:
            self.logger.error(f"Error loading feature data: {str(e)}")
            raise

    def perform_statistical_tests(self, controls_features, irbd_features):
        """
        Perform comprehensive statistical analysis comparing features between groups.

        WHAT THIS FUNCTION DOES:
        1. Tests each of the 1024 features for normality in both groups
        2. Chooses appropriate statistical test based on normality results
        3. Calculates effect sizes to quantify the magnitude of differences
        4. Applies multiple comparison correction to control false discoveries
        5. Identifies features with both statistical and practical significance

        STATISTICAL TESTS USED:
        - Shapiro-Wilk test: Checks if data follows a normal distribution
        - Independent t-test: Compares means when data is normally distributed
        - Mann-Whitney U test: Compares distributions when data is not normal
        - Bonferroni correction: Adjusts p-values for multiple comparisons

        EFFECT SIZE MEASURES:
        - Cohen's d: For t-tests, measures standardized difference between means
        - Rank-biserial correlation: For Mann-Whitney U, measures effect size

        Args:
            controls_features: Feature matrix for control participants (n_controls, 1024)
            irbd_features: Feature matrix for iRBD participants (n_irbd, 1024)

        Returns:
            pandas.DataFrame: Comprehensive results for each feature including:
                - Descriptive statistics for both groups
                - Statistical test results and p-values
                - Effect sizes and their interpretation
                - Multiple comparison corrected results
        """
        try:
            self.logger.info("Performing statistical tests for all features...")

            # Check that we have enough data for meaningful analysis
            if len(controls_features) < self.min_sample_size or len(irbd_features) < self.min_sample_size:
                raise ValueError(f"Insufficient sample size. Need at least {self.min_sample_size} "
                               f"participants per group. Got {len(controls_features)} controls, "
                               f"{len(irbd_features)} iRBD.")

            # Initialize results storage
            # We'll store results for each feature in these lists
            feature_results = {
                'feature_index': [],           # Which feature (0-1023)
                'feature_name': [],            # Human-readable feature name

                # Descriptive statistics for controls
                'control_mean': [],            # Average value in control group
                'control_std': [],             # Standard deviation in control group
                'control_median': [],          # Median value in control group
                'control_iqr': [],             # Interquartile range in control group

                # Descriptive statistics for iRBD
                'irbd_mean': [],               # Average value in iRBD group
                'irbd_std': [],                # Standard deviation in iRBD group
                'irbd_median': [],             # Median value in iRBD group
                'irbd_iqr': [],                # Interquartile range in iRBD group

                # Group comparison
                'mean_difference': [],         # Difference in means (iRBD - controls)
                'median_difference': [],       # Difference in medians (iRBD - controls)
                'percent_change': [],          # Percentage change from controls to iRBD

                # Normality testing
                'control_normal': [],          # Is control group normally distributed?
                'irbd_normal': [],             # Is iRBD group normally distributed?
                'control_shapiro_p': [],       # P-value from Shapiro-Wilk test (controls)
                'irbd_shapiro_p': [],          # P-value from Shapiro-Wilk test (iRBD)

                # Statistical testing
                'test_type': [],               # Which test was used (t-test or Mann-Whitney U)
                'test_statistic': [],          # Test statistic value
                'p_value': [],                 # Raw p-value from statistical test
                'p_value_corrected': [],       # Bonferroni corrected p-value
                'is_significant': [],          # Is the corrected p-value < 0.05?

                # Effect size analysis
                'effect_size': [],             # Effect size (Cohen's d or rank-biserial correlation)
                'effect_size_interpretation': [], # Small/Medium/Large effect
                'is_large_effect': [],         # Is effect size ≥ 0.5?

                # Combined significance
                'significant_and_large': []    # Both statistically significant AND large effect?
            }

            # Perform analysis for each feature
            self.logger.info(f"   Testing {self.feature_dim} features...")

            for feature_idx in range(self.feature_dim):
                # Extract feature values for both groups
                control_values = controls_features[:, feature_idx]  # All control participants, this feature
                irbd_values = irbd_features[:, feature_idx]         # All iRBD participants, this feature

                # Calculate descriptive statistics for controls
                control_mean = np.mean(control_values)
                control_std = np.std(control_values, ddof=1)  # Sample standard deviation (ddof=1)
                control_median = np.median(control_values)
                control_q75, control_q25 = np.percentile(control_values, [75, 25])
                control_iqr = control_q75 - control_q25

                # Calculate descriptive statistics for iRBD
                irbd_mean = np.mean(irbd_values)
                irbd_std = np.std(irbd_values, ddof=1)  # Sample standard deviation
                irbd_median = np.median(irbd_values)
                irbd_q75, irbd_q25 = np.percentile(irbd_values, [75, 25])
                irbd_iqr = irbd_q75 - irbd_q25

                # Calculate group differences
                mean_difference = irbd_mean - control_mean
                median_difference = irbd_median - control_median

                # Calculate percentage change (avoid division by zero)
                if abs(control_mean) > 1e-10:  # If control mean is not essentially zero
                    percent_change = (mean_difference / control_mean) * 100
                else:
                    percent_change = 0.0

                # Test for normality in both groups
                # We use Shapiro-Wilk test if sample size is appropriate (3-5000)
                control_normal = True
                irbd_normal = True
                control_shapiro_p = 1.0
                irbd_shapiro_p = 1.0

                # Test normality for controls
                if len(control_values) >= 3 and len(control_values) <= 5000:
                    try:
                        _, control_shapiro_p = shapiro(control_values)
                        control_normal = control_shapiro_p > self.normality_alpha
                    except:
                        # If Shapiro-Wilk fails, assume non-normal
                        control_normal = False
                        control_shapiro_p = 0.0

                # Test normality for iRBD
                if len(irbd_values) >= 3 and len(irbd_values) <= 5000:
                    try:
                        _, irbd_shapiro_p = shapiro(irbd_values)
                        irbd_normal = irbd_shapiro_p > self.normality_alpha
                    except:
                        # If Shapiro-Wilk fails, assume non-normal
                        irbd_normal = False
                        irbd_shapiro_p = 0.0

                # Choose appropriate statistical test based on normality
                if control_normal and irbd_normal:
                    # Both groups are normal - use independent t-test
                    # T-test compares the means of two groups
                    test_stat, p_value = ttest_ind(control_values, irbd_values, equal_var=False)
                    test_type = 'Independent t-test'

                    # Calculate Cohen's d for effect size
                    # Cohen's d measures the standardized difference between two means
                    pooled_std = np.sqrt(((len(control_values) - 1) * control_std**2 + 
                                         (len(irbd_values) - 1) * irbd_std**2) / 
                                        (len(control_values) + len(irbd_values) - 2))

                    if pooled_std > 0:
                        effect_size = mean_difference / pooled_std
                    else:
                        effect_size = 0.0

                else:
                    # At least one group is not normal - use Mann-Whitney U test
                    # Mann-Whitney U test compares the distributions of two groups
                    # It's more robust and doesn't assume normal distributions
                    test_stat, p_value = mannwhitneyu(control_values, irbd_values, 
                                                     alternative='two-sided')
                    test_type = 'Mann-Whitney U'

                    # Calculate rank-biserial correlation for effect size
                    # This is the appropriate effect size measure for Mann-Whitney U test
                    n1, n2 = len(control_values), len(irbd_values)
                    if (n1 * n2) > 0:
                        # Rank-biserial correlation formula
                        effect_size = (test_stat - (n1 * n2) / 2) / (n1 * n2)
                    else:
                        effect_size = 0.0

                # Interpret effect size magnitude
                abs_effect_size = abs(effect_size)
                if abs_effect_size < self.small_effect:
                    effect_interpretation = 'Negligible'
                elif abs_effect_size < self.medium_effect:
                    effect_interpretation = 'Small'
                elif abs_effect_size < self.large_effect:
                    effect_interpretation = 'Medium'
                else:
                    effect_interpretation = 'Large'

                # Determine if effect size is large enough to be practically significant
                is_large_effect = abs_effect_size >= self.medium_effect

                # Store all results for this feature
                feature_results['feature_index'].append(feature_idx)
                feature_results['feature_name'].append(f"Feature_{feature_idx:04d}")

                # Descriptive statistics
                feature_results['control_mean'].append(control_mean)
                feature_results['control_std'].append(control_std)
                feature_results['control_median'].append(control_median)
                feature_results['control_iqr'].append(control_iqr)
                feature_results['irbd_mean'].append(irbd_mean)
                feature_results['irbd_std'].append(irbd_std)
                feature_results['irbd_median'].append(irbd_median)
                feature_results['irbd_iqr'].append(irbd_iqr)

                # Group comparisons
                feature_results['mean_difference'].append(mean_difference)
                feature_results['median_difference'].append(median_difference)
                feature_results['percent_change'].append(percent_change)

                # Normality testing
                feature_results['control_normal'].append(control_normal)
                feature_results['irbd_normal'].append(irbd_normal)
                feature_results['control_shapiro_p'].append(control_shapiro_p)
                feature_results['irbd_shapiro_p'].append(irbd_shapiro_p)

                # Statistical testing
                feature_results['test_type'].append(test_type)
                feature_results['test_statistic'].append(test_stat)
                feature_results['p_value'].append(p_value)
                feature_results['effect_size'].append(effect_size)
                feature_results['effect_size_interpretation'].append(effect_interpretation)
                feature_results['is_large_effect'].append(is_large_effect)

                # Progress logging every 100 features
                if (feature_idx + 1) % 100 == 0:
                    self.logger.info(f"     - Processed {feature_idx + 1}/{self.feature_dim} features")

            # Apply multiple comparison correction (Bonferroni method)
            self.logger.info("   Applying Bonferroni correction for multiple comparisons...")

            # Extract all p-values for correction
            p_values = np.array(feature_results['p_value'])

            # Apply Bonferroni correction
            # This adjusts p-values to control the family-wise error rate
            rejected, p_corrected, _, _ = multipletests(p_values, alpha=self.alpha, method='bonferroni')

            # Add corrected results to our data
            feature_results['p_value_corrected'] = p_corrected.tolist()
            feature_results['is_significant'] = rejected.tolist()

            # Identify features with both significance and large effect
            both_sig_and_large = [sig and large for sig, large in 
                                 zip(feature_results['is_significant'], 
                                     feature_results['is_large_effect'])]
            feature_results['significant_and_large'] = both_sig_and_large

            # Calculate summary statistics
            n_significant = np.sum(rejected)
            n_large_effect = np.sum(feature_results['is_large_effect'])
            n_both = np.sum(both_sig_and_large)
            n_normal = np.sum([c and i for c, i in zip(feature_results['control_normal'], 
                                                      feature_results['irbd_normal'])])
            n_non_normal = self.feature_dim - n_normal

            # Update global statistics
            self.stats['total_features_tested'] = self.feature_dim
            self.stats['normal_features'] = n_normal
            self.stats['non_normal_features'] = n_non_normal
            self.stats['significant_features'] = n_significant
            self.stats['large_effect_features'] = n_large_effect
            self.stats['both_sig_and_large'] = n_both
            self.stats['effect_size_distribution'] = feature_results['effect_size'].copy()
            self.stats['p_value_distribution'] = feature_results['p_value'].copy()

            # Log summary results
            self.logger.info(f" Statistical testing completed:")
            self.logger.info(f"   - Total features tested: {self.feature_dim}")
            self.logger.info(f"   - Normal distributions: {n_normal} features")
            self.logger.info(f"   - Non-normal distributions: {n_non_normal} features")
            self.logger.info(f"   - Significant features (Bonferroni): {n_significant}")
            self.logger.info(f"   - Large effect features (|d|≥{self.medium_effect}): {n_large_effect}")
            self.logger.info(f"   - Both significant AND large effect: {n_both}")

            # Convert to DataFrame for easier handling and analysis
            results_df = pd.DataFrame(feature_results)

            # Sort by effect size (descending) to identify most important features
            results_df = results_df.sort_values('effect_size', key=abs, ascending=False)

            # Store top features for summary
            top_features = results_df.head(20)  # Top 20 features by effect size
            self.stats['top_features'] = top_features[['feature_index', 'effect_size', 
                                                      'p_value_corrected', 'test_type']].to_dict('records')

            return results_df

        except Exception as e:
            self.logger.error(f"Error in statistical testing: {str(e)}")
            raise

    def save_analysis_results(self, results_df, controls_features, irbd_features, participant_info):
        """
        Save all analysis results to files for later use and documentation.

        Args:
            results_df: DataFrame with statistical analysis results
            controls_features: Control group feature matrix
            irbd_features: iRBD group feature matrix
            participant_info: Dictionary with participant information
        """
        try:
            self.logger.info("Saving analysis results...")

            # Save detailed statistical results
            detailed_results_path = self.stats_results_dir / "detailed_statistical_results.csv"
            results_df.to_csv(detailed_results_path, index=False)
            self.logger.info(f"   - Detailed results saved: {detailed_results_path}")

            # Save significant features only
            significant_features = results_df[results_df['is_significant']]
            if len(significant_features) > 0:
                sig_results_path = self.stats_results_dir / "significant_features.csv"
                significant_features.to_csv(sig_results_path, index=False)
                self.logger.info(f"   - Significant features saved: {sig_results_path}")

            # Save features with large effects
            large_effect_features = results_df[results_df['is_large_effect']]
            if len(large_effect_features) > 0:
                large_effect_path = self.stats_results_dir / "large_effect_features.csv"
                large_effect_features.to_csv(large_effect_path, index=False)
                self.logger.info(f"   - Large effect features saved: {large_effect_path}")

            # Save features that are both significant and have large effects
            both_features = results_df[results_df['significant_and_large']]
            if len(both_features) > 0:
                both_path = self.stats_results_dir / "significant_and_large_effect_features.csv"
                both_features.to_csv(both_path, index=False)
                self.logger.info(f"   - Significant + large effect features saved: {both_path}")

            # Save comprehensive analysis summary
            summary = {
                'analysis_date': datetime.now().isoformat(),
                'dataset_info': {
                    'total_participants': self.stats['total_participants'],
                    'controls_count': self.stats['controls_count'],
                    'irbd_count': self.stats['irbd_count'],
                    'total_nights': self.stats['total_nights'],
                    'aggregation_method': self.aggregation_method
                },
                'statistical_parameters': {
                    'alpha': self.alpha,
                    'bonferroni_alpha': self.bonferroni_alpha,
                    'effect_size_threshold': self.medium_effect,
                    'normality_alpha': self.normality_alpha,
                    'min_sample_size': self.min_sample_size
                },
                'results_summary': {
                    'total_features_tested': self.stats['total_features_tested'],
                    'normal_features': self.stats['normal_features'],
                    'non_normal_features': self.stats['non_normal_features'],
                    'significant_features': self.stats['significant_features'],
                    'large_effect_features': self.stats['large_effect_features'],
                    'both_sig_and_large': self.stats['both_sig_and_large'],
                    'significance_rate': self.stats['significant_features'] / self.stats['total_features_tested'] * 100,
                    'large_effect_rate': self.stats['large_effect_features'] / self.stats['total_features_tested'] * 100
                },
                'top_features': self.stats['top_features'][:10],  # Top 10 features
                'participant_info': participant_info
            }

            def convert_numpy_types(obj):
                """Convert numpy types to native Python types for JSON serialization"""
                if isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            summary_path = self.stats_results_dir / "analysis_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(convert_numpy_types(summary), f, indent=2)

            # Save raw feature matrices for potential future analysis
            raw_data_path = self.stats_results_dir / "raw_feature_matrices.npz"
            np.savez_compressed(raw_data_path,
                              controls_features=controls_features,
                              irbd_features=irbd_features,
                              controls_ids=participant_info['controls'],
                              irbd_ids=participant_info['irbd'])
            self.logger.info(f"   - Raw feature matrices saved: {raw_data_path}")

        except Exception as e:
            self.logger.error(f"Error saving analysis results: {str(e)}")
            raise

    def create_comprehensive_visualizations(self, results_df, controls_features, irbd_features):
        """
        Create comprehensive visualizations for the statistical analysis results.

        Args:
            results_df: DataFrame with statistical analysis results
            controls_features: Control group feature matrix
            irbd_features: iRBD group feature matrix
        """
        try:
            self.logger.info("Creating comprehensive visualizations...")

            # Create output directory if it doesn't exist
            self.plots_dir.mkdir(parents=True, exist_ok=True)

            # 1. Volcano plot: Effect size vs Statistical significance
            plt.figure(figsize=(12, 8))

            # Extract data for plotting
            effect_sizes = results_df['effect_size'].values
            p_values_log = -np.log10(results_df['p_value_corrected'].values + 1e-300)  # Add small value to avoid log(0)

            # Color points based on significance and effect size
            colors = []
            for _, row in results_df.iterrows():
                if row['significant_and_large']:
                    colors.append('red')  # Both significant and large effect
                elif row['is_significant']:
                    colors.append('orange')  # Significant only
                elif row['is_large_effect']:
                    colors.append('blue')  # Large effect only
                else:
                    colors.append('lightgray')  # Neither

            # Create scatter plot
            plt.scatter(effect_sizes, p_values_log, c=colors, alpha=0.6, s=30)

            # Add significance and effect size thresholds
            plt.axhline(y=-np.log10(self.bonferroni_alpha), color='red', linestyle='--', 
                       label=f'Bonferroni threshold (α={self.bonferroni_alpha:.2e})')
            plt.axvline(x=self.medium_effect, color='blue', linestyle='--', 
                       label=f'Effect size threshold (|d|={self.medium_effect})')
            plt.axvline(x=-self.medium_effect, color='blue', linestyle='--')

            plt.xlabel('Effect Size (Cohen\'s d or rank-biserial correlation)', fontsize=12)
            plt.ylabel('-log₁₀(Corrected p-value)', fontsize=12)
            plt.title('Feature Significance Analysis - iRBD vs Controls\nVolcano Plot', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add text annotations for quadrants
            plt.text(0.7, max(p_values_log)*0.9, 'High Effect\nSignificant', 
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
            plt.text(-0.7, max(p_values_log)*0.9, 'High Effect\nSignificant', 
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

            volcano_path = self.plots_dir / "feature_volcano_plot.png"
            plt.savefig(volcano_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"   - Volcano plot saved: {volcano_path}")

            # 2. Top discriminative features
            top_features = results_df.head(20)  # Top 20 by effect size

            plt.figure(figsize=(12, 10))
            y_pos = np.arange(len(top_features))

            # Create horizontal bar plot
            colors = ['red' if both else 'orange' if sig else 'lightblue' 
                     for both, sig in zip(top_features['significant_and_large'], 
                                         top_features['is_significant'])]

            bars = plt.barh(y_pos, top_features['effect_size'].abs(), color=colors)

            plt.yticks(y_pos, [f'Feature {idx}' for idx in top_features['feature_index']])
            plt.xlabel('|Effect Size|', fontsize=12)
            plt.title('Top 20 Most Discriminative Features', fontsize=14, fontweight='bold')
            plt.axvline(x=self.medium_effect, color='blue', linestyle='--', 
                       label=f'Medium effect threshold (|d|={self.medium_effect})')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='x')

            # Add value labels
            for i, (bar, effect_size) in enumerate(zip(bars, top_features['effect_size'].abs())):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{effect_size:.3f}', va='center', fontsize=9)

            plt.tight_layout()

            top_features_path = self.plots_dir / "top_discriminative_features.png"
            plt.savefig(top_features_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"   - Top features plot saved: {top_features_path}")

            # 3. Statistical analysis summary dashboard
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # Subplot 1: Feature significance summary
            categories = ['Total\nFeatures', 'Significant\n(Bonferroni)', 'Large Effect\n(|d|≥0.5)', 
                         'Both Sig.\n& Large']
            counts = [
                self.stats['total_features_tested'],
                self.stats['significant_features'],
                self.stats['large_effect_features'],
                self.stats['both_sig_and_large']
            ]

            bars1 = ax1.bar(categories, counts, color=['lightgray', 'orange', 'lightblue', 'red'])
            ax1.set_title('Feature Analysis Summary', fontweight='bold')
            ax1.set_ylabel('Number of Features')

            # Add value labels on bars
            for bar, count in zip(bars1, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                        str(count), ha='center', va='bottom', fontweight='bold')

            # Subplot 2: Effect size distribution
            ax2.hist(results_df['effect_size'].abs(), bins=50, alpha=0.7, color='skyblue', 
                    edgecolor='black')
            ax2.axvline(x=self.medium_effect, color='red', linestyle='--', 
                       label=f'Medium effect (|d|={self.medium_effect})')
            ax2.set_title('Distribution of |Effect Sizes|', fontweight='bold')
            ax2.set_xlabel('|Effect Size|')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Subplot 3: P-value distribution
            ax3.hist(-np.log10(results_df['p_value'] + 1e-300), bins=50, alpha=0.7, 
                    color='lightcoral', edgecolor='black')
            ax3.axvline(x=-np.log10(self.bonferroni_alpha), color='red', linestyle='--', 
                       label=f'Bonferroni threshold')
            ax3.set_title('Distribution of -log₁₀(p-values)', fontweight='bold')
            ax3.set_xlabel('-log₁₀(p-value)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Subplot 4: Test type usage
            test_counts = results_df['test_type'].value_counts()
            colors = ['lightgreen', 'lightcoral']
            wedges, texts, autotexts = ax4.pie(test_counts.values, labels=test_counts.index, 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            ax4.set_title('Statistical Tests Used', fontweight='bold')

            plt.suptitle('Statistical Analysis Dashboard - iRBD vs Controls', fontsize=16, fontweight='bold')
            plt.tight_layout()

            dashboard_path = self.plots_dir / "statistical_analysis_dashboard.png"
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"   - Analysis dashboard saved: {dashboard_path}")

            # 4. Group comparison for top features
            top_significant = results_df[results_df['significant_and_large']].head(6)

            if len(top_significant) > 0:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()

                for i, (_, feature_row) in enumerate(top_significant.iterrows()):
                    if i >= 6:
                        break

                    feature_idx = feature_row['feature_index']

                    # Get feature values for both groups
                    control_values = controls_features[:, feature_idx]
                    irbd_values = irbd_features[:, feature_idx]

                    # Create box plot
                    data_to_plot = [control_values, irbd_values]
                    box_plot = axes[i].boxplot(data_to_plot, labels=['Controls', 'iRBD'], 
                                              patch_artist=True)

                    # Color the boxes
                    box_plot['boxes'][0].set_facecolor('lightblue')
                    box_plot['boxes'][1].set_facecolor('lightcoral')

                    axes[i].set_title(f'Feature {feature_idx}\n'
                                     f'Effect size: {feature_row["effect_size"]:.3f}\n'
                                     f'p-value: {feature_row["p_value_corrected"]:.2e}')
                    axes[i].set_ylabel('Feature Value')
                    axes[i].grid(True, alpha=0.3)

                # Hide unused subplots
                for i in range(len(top_significant), 6):
                    axes[i].set_visible(False)

                plt.suptitle('Top Discriminative Features - Group Comparisons', fontsize=16, fontweight='bold')
                plt.tight_layout()

                group_comparison_path = self.plots_dir / "top_features_group_comparison.png"
                plt.savefig(group_comparison_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"   - Group comparison plot saved: {group_comparison_path}")

            # 5. Feature space visualization using t-SNE
            if len(controls_features) > 0 and len(irbd_features) > 0:
                self.logger.info("   Creating t-SNE visualization of feature space...")

                # Combine all features
                all_features = np.vstack([controls_features, irbd_features])
                all_labels = ['Controls'] * len(controls_features) + ['iRBD'] * len(irbd_features)

                # Standardize features for t-SNE
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(all_features)

                # Apply t-SNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)-1))
                features_2d = tsne.fit_transform(features_scaled)

                # Create t-SNE plot
                plt.figure(figsize=(10, 8))

                # Plot controls
                control_indices = np.array(all_labels) == 'Controls'
                plt.scatter(features_2d[control_indices, 0], features_2d[control_indices, 1], 
                           c='blue', alpha=0.6, s=50, label='Controls')

                # Plot iRBD
                irbd_indices = np.array(all_labels) == 'iRBD'
                plt.scatter(features_2d[irbd_indices, 0], features_2d[irbd_indices, 1], 
                           c='red', alpha=0.6, s=50, label='iRBD')

                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.title('t-SNE Visualization of Feature Space\niRBD vs Controls', fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)

                tsne_path = self.plots_dir / "feature_space_tsne.png"
                plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"   - t-SNE visualization saved: {tsne_path}")

        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            raise

    def run_statistical_analysis(self):
        """
        Main function to run the complete statistical analysis pipeline.
        """
        try:
            self.logger.info("Running complete statistical analysis pipeline...")

            # Record start time
            analysis_start_time = datetime.now()

            # Load and aggregate feature data
            controls_features, irbd_features, participant_info = self.load_and_aggregate_features()

            # Perform statistical tests
            results_df = self.perform_statistical_tests(controls_features, irbd_features)

            # Save results
            self.save_analysis_results(results_df, controls_features, irbd_features, participant_info)

            # Create visualizations
            self.create_comprehensive_visualizations(results_df, controls_features, irbd_features)

            # Calculate analysis time
            analysis_time = (datetime.now() - analysis_start_time).total_seconds()
            self.stats['analysis_time'] = analysis_time

            # Show final results
            self.print_final_statistics()

            self.logger.info("STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY!")

        except Exception as e:
            self.logger.error(f"STATISTICAL ANALYSIS FAILED: {str(e)}")
            raise

    def print_final_statistics(self):
        """
        Print comprehensive summary of the statistical analysis results.
        """
        # Print header
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"STATISTICAL ANALYSIS COMPLETED")
        self.logger.info(f"{'='*70}")

        # Dataset statistics
        self.logger.info(f"Dataset Statistics:")
        self.logger.info(f"  - Total participants: {self.stats['total_participants']}")
        self.logger.info(f"  - Controls: {self.stats['controls_count']}")
        self.logger.info(f"  - iRBD: {self.stats['irbd_count']}")
        self.logger.info(f"  - Total nights: {self.stats['total_nights']}")
        self.logger.info(f"  - Aggregation method: {self.aggregation_method}")
        self.logger.info("")

        # Statistical analysis results
        self.logger.info(f"Feature Analysis Results:")
        self.logger.info(f"  - Total features tested: {self.stats['total_features_tested']}")
        self.logger.info(f"  - Normal distributions: {self.stats['normal_features']}")
        self.logger.info(f"  - Non-normal distributions: {self.stats['non_normal_features']}")
        self.logger.info(f"  - Significant features (Bonferroni): {self.stats['significant_features']}")
        self.logger.info(f"  - Large effect features (|d|≥{self.medium_effect}): {self.stats['large_effect_features']}")
        self.logger.info(f"  - Both significant AND large effect: {self.stats['both_sig_and_large']}")

        if self.stats['total_features_tested'] > 0:
            sig_rate = self.stats['significant_features'] / self.stats['total_features_tested'] * 100
            effect_rate = self.stats['large_effect_features'] / self.stats['total_features_tested'] * 100
            both_rate = self.stats['both_sig_and_large'] / self.stats['total_features_tested'] * 100
            self.logger.info(f"  - Significance rate: {sig_rate:.2f}%")
            self.logger.info(f"  - Large effect rate: {effect_rate:.2f}%")
            self.logger.info(f"  - Both significant & large effect rate: {both_rate:.2f}%")

        self.logger.info("")

        # Statistical parameters used
        self.logger.info(f"Statistical Parameters:")
        self.logger.info(f"  - Significance level (α): {self.alpha}")
        self.logger.info(f"  - Bonferroni corrected α: {self.bonferroni_alpha:.2e}")
        self.logger.info(f"  - Effect size threshold: {self.medium_effect}")
        self.logger.info(f"  - Normality test α: {self.normality_alpha}")
        self.logger.info("")

        # Analysis performance
        self.logger.info(f"Analysis Performance:")
        self.logger.info(f"  - Total analysis time: {self.stats['analysis_time']:.1f} seconds")
        self.logger.info("")

        # Top discriminative features
        if self.stats['top_features']:
            self.logger.info(f"Top 5 Most Discriminative Features:")
            for i, feature in enumerate(self.stats['top_features'][:5], 1):
                self.logger.info(f"  {i}. Feature {feature['feature_index']}: "
                               f"|d|={abs(feature['effect_size']):.3f}, "
                               f"p={feature['p_value_corrected']:.2e}, "
                               f"test={feature['test_type']}")
            self.logger.info("")

        # Clinical interpretation
        self.logger.info(f"Clinical Interpretation:")
        if self.stats['both_sig_and_large'] > 0:
            self.logger.info(f"  Found {self.stats['both_sig_and_large']} features with both statistical")
            self.logger.info(f"     significance and large effect sizes")
            self.logger.info(f"  These features represent movement patterns that reliably")
            self.logger.info(f"     distinguish iRBD patients from healthy controls")
            self.logger.info(f"  Results support the use of SSL-Wearables features for iRBD detection")
        else:
            self.logger.info(f"   No features found with both statistical significance and large effects")
            self.logger.info(f"   May need larger sample size or different analysis approach")

        self.logger.info("")

        # Final success message
        if self.stats['total_features_tested'] > 0:
            self.logger.info("Statistical analysis pipeline completed successfully!")
            self.logger.info("Complete analysis ready for thesis writing and publication")
        else:
            self.logger.error("Statistical analysis pipeline failed!")


# In[ ]:


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """
    Main function that runs when the script is executed directly.
    Creates a FeatureStatisticalAnalysis object and runs the entire analysis process.
    """
    try:
        # Create and run the statistical analysis pipeline
        pipeline = FeatureStatisticalAnalysis()
        pipeline.run_statistical_analysis()

    except KeyboardInterrupt:
        print("\n Statistical analysis interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n Statistical analysis failed with error: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

# =============================================================================
# SCRIPT EXECUTION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

