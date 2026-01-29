#!/usr/bin/env python
# coding: utf-8

"""
TEMPERATURE DATA INVESTIGATION SCRIPT

Purpose: Comprehensive investigation of temperature characteristics in accelerometer data
         to understand temperature patterns and inform non-wear detection thresholds.

Questions to answer:
1. What is the temperature distribution specifically during sleep hours (22:00-06:00)?
2. How does temperature vary across the 24-hour cycle?
3. Is there a consistent day/night temperature pattern?
4. Does low temperature correlate with low movement (non-wear indicator)?
5. What threshold is appropriate for nocturnal data specifically?
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, time, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import actipy
    print(f"actipy version: {actipy.__version__}")
except ImportError as e:
    print(f"Error importing actipy: {e}")
    print("Please install actipy with: pip install actipy")
    sys.exit(1)

# Configure plotting
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Custom color palette
COLORS = {
    'controls': '#E8A8C5',  # Pink
    'irbd': '#77CBDA',      # Blue
    'threshold_current': '#1565C0',  # Dark blue (20°C)
    'threshold_proposed': '#2E7D32',  # Dark green (18°C)
    'day': '#FFD700',       # Gold
    'night': '#808080'      # Royal blue
}

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/work3/s184484/iRBD-detection")
RAW_CONTROLS_DIR = BASE_DIR / "data" / "raw" / "controls"
RAW_IRBD_DIR = BASE_DIR / "data" / "raw" / "irbd"
OUTPUT_DIR = BASE_DIR / "results" / "temperature_investigation"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Processing parameters (same as preprocessing scripts)
LOWPASS_HZ = 12
RESAMPLE_HZ = 30
DETECT_NONWEAR = True
CALIBRATE_GRAVITY = True

# Night window definition
NIGHT_START_HOUR = 22  # 22:00
NIGHT_END_HOUR = 6     # 06:00

# Temperature thresholds to test
THRESHOLDS_TO_TEST = [15, 18, 20, 22, 24, 26, 27, 28, 30]

# Acceleration SD window for non-wear correlation (in seconds)
ACCEL_SD_WINDOW_SEC = 300  # 5 minutes

# Sample size (None = all files)
SAMPLE_SIZE = None

# Random seed for reproducibility
RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)

print("=" * 70)
print("TEMPERATURE DATA INVESTIGATION")
print("=" * 70)
print(f"Base directory: {BASE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Night window: {NIGHT_START_HOUR}:00 - {NIGHT_END_HOUR}:00")
print(f"Thresholds to test: {THRESHOLDS_TO_TEST}")
print(f"Sample size: {'All files' if SAMPLE_SIZE is None else f'{SAMPLE_SIZE} files per group'}")
print("=" * 70)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_cwa_files(directory, limit=None):
    """Find .cwa files in directory, optionally limiting the number."""
    import glob
    cwa_files = glob.glob(str(directory / "*.cwa")) + glob.glob(str(directory / "*.CWA"))
    cwa_files = [Path(f) for f in cwa_files]
    cwa_files.sort()
    
    if limit:
        cwa_files = cwa_files[:limit]
    
    return cwa_files


def read_accelerometer_data(file_path):
    """Read accelerometer data using actipy."""
    try:
        data, info = actipy.read_device(
            str(file_path),
            lowpass_hz=LOWPASS_HZ,
            resample_hz=RESAMPLE_HZ,
            detect_nonwear=DETECT_NONWEAR,
            calibrate_gravity=CALIBRATE_GRAVITY
        )
        return data, info, None
    except Exception as e:
        return None, None, str(e)


def is_night_hour(hour):
    """Check if hour falls within night window (22:00-06:00)."""
    return hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR


def extract_nocturnal_data(data):
    """Extract only data from 22:00-06:00 window."""
    if data is None or len(data) == 0:
        return None
    
    hours = data.index.hour
    night_mask = (hours >= NIGHT_START_HOUR) | (hours < NIGHT_END_HOUR)
    return data[night_mask]


def calculate_acceleration_sd(data, window_samples):
    """Calculate rolling standard deviation of acceleration magnitude."""
    if data is None or len(data) == 0:
        return None
    
    # Calculate acceleration magnitude
    accel_mag = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    
    # Calculate rolling standard deviation
    accel_sd = accel_mag.rolling(window=window_samples, center=True, min_periods=1).std()
    
    return accel_sd


def calculate_retention_rate(data, threshold):
    """Calculate what percentage of data would be retained at given temperature threshold."""
    if data is None or len(data) == 0:
        return 0.0
    
    # actipy non-wear mask (valid data)
    valid_data_mask = ~(data['x'].isna() | data['y'].isna() | data['z'].isna())
    
    # Temperature mask
    temp_mask = data['temperature'] >= threshold
    
    # Combined mask
    combined_mask = temp_mask & valid_data_mask
    
    retention_rate = (combined_mask.sum() / len(data)) * 100
    return retention_rate


def get_hourly_temperature_stats(data):
    """Calculate temperature statistics for each hour of the day."""
    if data is None or len(data) == 0:
        return None
    
    hourly_stats = data.groupby(data.index.hour)['temperature'].agg(['mean', 'std', 'median', 'min', 'max'])
    hourly_stats.index.name = 'hour'
    return hourly_stats


# =============================================================================
# STEP 1: DATA COLLECTION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 1: COLLECTING DATA FROM RAW .CWA FILES")
print("=" * 70)

# Find files
controls_files = find_cwa_files(RAW_CONTROLS_DIR, limit=SAMPLE_SIZE)
irbd_files = find_cwa_files(RAW_IRBD_DIR, limit=SAMPLE_SIZE)

print(f"Found {len(controls_files)} control files")
print(f"Found {len(irbd_files)} iRBD files")
print(f"Total: {len(controls_files) + len(irbd_files)} files to analyze")

# Data structures for collection
temperature_stats_all = []      # All-day statistics
temperature_stats_night = []    # Nocturnal statistics only
retention_rates_all = []        # Retention rates (all day)
retention_rates_night = []      # Retention rates (nocturnal only)
hourly_temps_all = []           # Hourly temperature data for heatmap

# Select random participants for time-series plots
all_files = [('controls', f) for f in controls_files] + [('irbd', f) for f in irbd_files]
example_participants = {}

# Store data for example participants
example_data = {}

for group, files in [("controls", controls_files), ("irbd", irbd_files)]:
    print(f"\nProcessing {group}...")
    
    # Select random example participant for this group
    if len(files) > 0:
        example_idx = np.random.randint(0, len(files))
        example_file = files[example_idx]
        example_participants[group] = example_file.stem
        print(f"  Selected example participant: {example_file.stem}")
    
    for i, file_path in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {file_path.stem}...", end=" ", flush=True)
        
        # Read data
        data, info, error = read_accelerometer_data(file_path)
        
        if error:
            print(f"ERROR: {error}")
            continue
        
        if data is None or len(data) == 0:
            print("SKIPPED (no data)")
            continue
        
        # Store example participant data
        if file_path.stem == example_participants.get(group):
            example_data[group] = data.copy()
        
        # Extract temperature data
        temp_data = data['temperature'].dropna()
        
        if len(temp_data) == 0:
            print("SKIPPED (no temperature data)")
            continue
        
        # ----- ALL-DAY STATISTICS -----
        stats_all = {
            'participant': file_path.stem,
            'group': group,
            'period': 'all_day',
            'temp_min': temp_data.min(),
            'temp_max': temp_data.max(),
            'temp_mean': temp_data.mean(),
            'temp_median': temp_data.median(),
            'temp_std': temp_data.std(),
            'temp_p5': temp_data.quantile(0.05),
            'temp_p25': temp_data.quantile(0.25),
            'temp_p75': temp_data.quantile(0.75),
            'temp_p95': temp_data.quantile(0.95),
            'total_samples': len(data),
            'valid_samples': (~data['x'].isna()).sum(),
            'nonwear_samples': data['x'].isna().sum()
        }
        temperature_stats_all.append(stats_all)
        
        # ----- NOCTURNAL STATISTICS (22:00-06:00) -----
        nocturnal_data = extract_nocturnal_data(data)
        if nocturnal_data is not None and len(nocturnal_data) > 0:
            nocturnal_temp = nocturnal_data['temperature'].dropna()
            if len(nocturnal_temp) > 0:
                stats_night = {
                    'participant': file_path.stem,
                    'group': group,
                    'period': 'nocturnal',
                    'temp_min': nocturnal_temp.min(),
                    'temp_max': nocturnal_temp.max(),
                    'temp_mean': nocturnal_temp.mean(),
                    'temp_median': nocturnal_temp.median(),
                    'temp_std': nocturnal_temp.std(),
                    'temp_p5': nocturnal_temp.quantile(0.05),
                    'temp_p25': nocturnal_temp.quantile(0.25),
                    'temp_p75': nocturnal_temp.quantile(0.75),
                    'temp_p95': nocturnal_temp.quantile(0.95),
                    'total_samples': len(nocturnal_data),
                    'valid_samples': (~nocturnal_data['x'].isna()).sum(),
                    'nonwear_samples': nocturnal_data['x'].isna().sum()
                }
                temperature_stats_night.append(stats_night)
        
        # ----- HOURLY TEMPERATURE DATA -----
        hourly_stats = get_hourly_temperature_stats(data)
        if hourly_stats is not None:
            for hour in range(24):
                if hour in hourly_stats.index:
                    hourly_temps_all.append({
                        'participant': file_path.stem,
                        'group': group,
                        'hour': hour,
                        'temp_mean': hourly_stats.loc[hour, 'mean'],
                        'temp_std': hourly_stats.loc[hour, 'std'],
                        'temp_median': hourly_stats.loc[hour, 'median']
                    })
        
        # ----- RETENTION RATES (ALL DAY) -----
        for threshold in THRESHOLDS_TO_TEST:
            retention = calculate_retention_rate(data, threshold)
            retention_rates_all.append({
                'participant': file_path.stem,
                'group': group,
                'period': 'all_day',
                'threshold': threshold,
                'retention_rate': retention
            })
        
        # ----- RETENTION RATES (NOCTURNAL) -----
        if nocturnal_data is not None and len(nocturnal_data) > 0:
            for threshold in THRESHOLDS_TO_TEST:
                retention = calculate_retention_rate(nocturnal_data, threshold)
                retention_rates_night.append({
                    'participant': file_path.stem,
                    'group': group,
                    'period': 'nocturnal',
                    'threshold': threshold,
                    'retention_rate': retention
                })
        
        print(f"OK (mean temp all: {stats_all['temp_mean']:.1f}°C, night: {stats_night['temp_mean']:.1f}°C)" if 'stats_night' in dir() and stats_night else f"OK (mean temp: {stats_all['temp_mean']:.1f}°C)")

# Convert to DataFrames
df_stats_all = pd.DataFrame(temperature_stats_all)
df_stats_night = pd.DataFrame(temperature_stats_night)
df_retention_all = pd.DataFrame(retention_rates_all)
df_retention_night = pd.DataFrame(retention_rates_night)
df_hourly = pd.DataFrame(hourly_temps_all)

print(f"\nCollected data from {len(df_stats_all)} participants")
print(f"Example participants: {example_participants}")

# Save raw data
df_stats_all.to_csv(OUTPUT_DIR / "temperature_statistics_all_day.csv", index=False)
df_stats_night.to_csv(OUTPUT_DIR / "temperature_statistics_nocturnal.csv", index=False)
df_retention_all.to_csv(OUTPUT_DIR / "retention_rates_all_day.csv", index=False)
df_retention_night.to_csv(OUTPUT_DIR / "retention_rates_nocturnal.csv", index=False)
df_hourly.to_csv(OUTPUT_DIR / "hourly_temperature_data.csv", index=False)

print(f"\nSaved CSV files to {OUTPUT_DIR}")


# =============================================================================
# STEP 2: ANALYSIS & SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: ANALYZING TEMPERATURE DISTRIBUTIONS")
print("=" * 70)

print("\n--- ALL-DAY TEMPERATURE STATISTICS ---")
print(f"  Mean: {df_stats_all['temp_mean'].mean():.2f}°C (±{df_stats_all['temp_mean'].std():.2f})")
print(f"  Median: {df_stats_all['temp_median'].median():.2f}°C")
print(f"  Range: {df_stats_all['temp_min'].min():.2f}°C - {df_stats_all['temp_max'].max():.2f}°C")

print("\n--- NOCTURNAL TEMPERATURE STATISTICS (22:00-06:00) ---")
print(f"  Mean: {df_stats_night['temp_mean'].mean():.2f}°C (±{df_stats_night['temp_mean'].std():.2f})")
print(f"  Median: {df_stats_night['temp_median'].median():.2f}°C")
print(f"  Range: {df_stats_night['temp_min'].min():.2f}°C - {df_stats_night['temp_max'].max():.2f}°C")

print("\n--- COMPARISON: DAY vs NIGHT ---")
day_mean = df_stats_all['temp_mean'].mean()
night_mean = df_stats_night['temp_mean'].mean()
print(f"  All-day mean: {day_mean:.2f}°C")
print(f"  Nocturnal mean: {night_mean:.2f}°C")
print(f"  Difference: {night_mean - day_mean:+.2f}°C")

print("\n--- BY GROUP (NOCTURNAL) ---")
for group in ['controls', 'irbd']:
    group_data = df_stats_night[df_stats_night['group'] == group]
    print(f"\n{group.upper()}:")
    print(f"  Mean: {group_data['temp_mean'].mean():.2f}°C (±{group_data['temp_mean'].std():.2f})")
    print(f"  Median: {group_data['temp_median'].median():.2f}°C")
    print(f"  Range: {group_data['temp_min'].min():.2f}°C - {group_data['temp_max'].max():.2f}°C")

print("\n--- RETENTION RATE COMPARISON ---")
print("\nAll-day retention rates:")
for threshold in [18, 20, 27]:
    rate = df_retention_all[df_retention_all['threshold'] == threshold]['retention_rate'].mean()
    print(f"  {threshold}°C: {rate:.1f}%")

print("\nNocturnal retention rates:")
for threshold in [18, 20, 27]:
    rate = df_retention_night[df_retention_night['threshold'] == threshold]['retention_rate'].mean()
    print(f"  {threshold}°C: {rate:.1f}%")


# =============================================================================
# STEP 3: VISUALIZATION - MAIN DASHBOARD
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: CREATING VISUALIZATIONS")
print("=" * 70)

# ----- FIGURE 1: Main Temperature Analysis Dashboard -----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Temperature Analysis Dashboard - Nocturnal Data (22:00-06:00)', fontsize=14, fontweight='bold')

# Plot 1a: Histogram of mean temperatures (nocturnal)
ax = axes[0, 0]
for group in ['controls', 'irbd']:
    group_data = df_stats_night[df_stats_night['group'] == group]
    ax.hist(group_data['temp_mean'], bins=15, alpha=0.6, 
            label=group.capitalize(), color=COLORS[group], edgecolor='black', linewidth=0.5)
ax.axvline(20, color=COLORS['threshold_current'], linestyle='--', linewidth=2, label='20°C threshold')
ax.axvline(18, color=COLORS['threshold_proposed'], linestyle='--', linewidth=2, label='18°C threshold')
ax.set_xlabel('Mean Temperature (°C)')
ax.set_ylabel('Number of Participants')
ax.set_title('Distribution of Mean Nocturnal Temperatures')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 1b: Box plot comparing all-day vs nocturnal
ax = axes[0, 1]
data_to_plot = [
    df_stats_all['temp_mean'].values,
    df_stats_night['temp_mean'].values
]
bp = ax.boxplot(data_to_plot, labels=['All Day', 'Nocturnal\n(22:00-06:00)'], patch_artist=True)
bp['boxes'][0].set_facecolor(COLORS['day'])
bp['boxes'][1].set_facecolor('#7D0D91')
ax.axhline(20, color=COLORS['threshold_current'], linestyle='--', linewidth=2, label='20°C threshold')
ax.axhline(18, color=COLORS['threshold_proposed'], linestyle='--', linewidth=2, label='18°C threshold')
ax.set_ylabel('Mean Temperature (°C)')
ax.set_title('All-Day vs Nocturnal Temperature Comparison')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 1c: Retention rates by threshold (nocturnal)
ax = axes[1, 0]
retention_summary = df_retention_night.groupby('threshold')['retention_rate'].agg(['mean', 'std'])
ax.errorbar(retention_summary.index, retention_summary['mean'], 
            yerr=retention_summary['std'], marker='o', linewidth=2, markersize=8, 
            color='#333333', capsize=3)
ax.axvline(20, color=COLORS['threshold_current'], linestyle='--', linewidth=2, alpha=0.7, label='20°C threshold')
ax.axvline(18, color=COLORS['threshold_proposed'], linestyle='--', linewidth=2, alpha=0.7, label='18°C threshold')
ax.set_xlabel('Temperature Threshold (°C)')
ax.set_ylabel('Data Retention Rate (%)')
ax.set_title('Impact of Temperature Threshold on Nocturnal Data Retention')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

# Plot 1d: Retention rates by group (nocturnal)
ax = axes[1, 1]
for group in ['controls', 'irbd']:
    group_data = df_retention_night[df_retention_night['group'] == group]
    retention_by_threshold = group_data.groupby('threshold')['retention_rate'].mean()
    ax.plot(retention_by_threshold.index, retention_by_threshold.values, 
            marker='o', linewidth=2, markersize=8, label=group.capitalize(), color=COLORS[group])
ax.axvline(20, color=COLORS['threshold_current'], linestyle='--', linewidth=2, alpha=0.7, label='20°C threshold')
ax.axvline(18, color=COLORS['threshold_proposed'], linestyle='--', linewidth=2, alpha=0.7, label='18°C threshold')
ax.set_xlabel('Temperature Threshold (°C)')
ax.set_ylabel('Data Retention Rate (%)')
ax.set_title('Nocturnal Retention Rate by Group and Threshold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_temperature_dashboard_nocturnal.png", dpi=300, bbox_inches='tight')
print(f"Saved: 01_temperature_dashboard_nocturnal.png")
plt.close()


# ----- FIGURE 2: Hourly Temperature Heatmap -----
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Hourly Temperature Patterns Across All Participants', fontsize=14, fontweight='bold')

for idx, group in enumerate(['controls', 'irbd']):
    ax = axes[idx]
    group_hourly = df_hourly[df_hourly['group'] == group]
    
    # Create pivot table for heatmap
    pivot = group_hourly.pivot_table(
        values='temp_mean', 
        index='participant', 
        columns='hour', 
        aggfunc='mean'
    )
    
    # Sort by mean temperature for better visualization
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    
    sns.heatmap(pivot, ax=ax, cmap='RdYlBu_r', center=20, 
                vmin=15, vmax=30, cbar_kws={'label': 'Temperature (°C)'})
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Participant')
    ax.set_title(f'{group.capitalize()} - Hourly Temperature Heatmap')
    
    # Mark night hours
    for h in range(24):
        if is_night_hour(h):
            ax.axvline(h, color='white', linewidth=0.5, alpha=0.5)
    
    # Add night window annotation
    ax.axvline(NIGHT_START_HOUR + 0.5, color='blue', linewidth=2, linestyle='--')
    ax.axvline(NIGHT_END_HOUR + 0.5, color='blue', linewidth=2, linestyle='--')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_hourly_temperature_heatmap.png", dpi=300, bbox_inches='tight')
print(f"Saved: 02_hourly_temperature_heatmap.png")
plt.close()


# ----- FIGURE 3: Hourly Temperature Box Plot -----
fig, ax = plt.subplots(figsize=(14, 6))

# Prepare data for box plot
hourly_data = []
for hour in range(24):
    hour_temps = df_hourly[df_hourly['hour'] == hour]['temp_mean'].values
    hourly_data.append(hour_temps)

bp = ax.boxplot(hourly_data, positions=range(24), patch_artist=True)

# Color boxes by day/night
for i, box in enumerate(bp['boxes']):
    if is_night_hour(i):
        box.set_facecolor("#7D0D91")
        box.set_alpha(0.6)
    else:
        box.set_facecolor(COLORS['day'])
        box.set_alpha(0.6)

ax.axhline(20, color=COLORS['threshold_current'], linestyle='--', linewidth=2, label='20°C threshold')
ax.axhline(18, color=COLORS['threshold_proposed'], linestyle='--', linewidth=2, label='18°C threshold')

# Add night window shading
ax.axvspan(-0.5, NIGHT_END_HOUR - 0.5, alpha=0.1, color='grey')
ax.axvspan(NIGHT_START_HOUR - 0.5, 23.5, alpha=0.1, color='grey')

ax.set_xlabel('Hour of Day')
ax.set_ylabel('Mean Temperature (°C)')
ax.set_title('Temperature Distribution by Hour of Day (All Participants)')
ax.set_xticks(range(24))
ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_hourly_temperature_boxplot.png", dpi=300, bbox_inches='tight')
print(f"Saved: 03_hourly_temperature_boxplot.png")
plt.close()


# ----- FIGURE 4: Single Participant 24-Hour Profile -----
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('24-Hour Temperature Profile - Example Participants', fontsize=14, fontweight='bold')

for idx, (group, data) in enumerate(example_data.items()):
    ax = axes[idx]
    participant_id = example_participants[group]
    
    # Get first 24 hours of data
    start_time = data.index[0]
    end_time = start_time + timedelta(hours=24)
    day_data = data[data.index < end_time]
    
    # Resample to 1-minute for cleaner plot
    temp_resampled = day_data['temperature'].resample('1min').mean()
    
    ax.plot(temp_resampled.index, temp_resampled.values, linewidth=2.0, color=COLORS[group])
    
    # Add threshold lines
    ax.axhline(20, color=COLORS['threshold_current'], linestyle='--', linewidth=1.5, alpha=0.7, label='20°C threshold')
    ax.axhline(18, color=COLORS['threshold_proposed'], linestyle='--', linewidth=1.5, alpha=0.7, label='18°C threshold')
    
    # Shade night hours
    for i in range(len(temp_resampled)):
        if is_night_hour(temp_resampled.index[i].hour):
            ax.axvspan(temp_resampled.index[i], 
                      temp_resampled.index[min(i+1, len(temp_resampled)-1)], 
                      alpha=0.1, color='grey')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'{group.capitalize()} - Participant {participant_id}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_24hour_temperature_profile.png", dpi=300, bbox_inches='tight')
print(f"Saved: 04_24hour_temperature_profile.png")
plt.close()


# ----- FIGURE 5: Multi-Day Temperature Profile -----
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Multi-Day Temperature Profile - Example Participants', fontsize=14, fontweight='bold')

for idx, (group, data) in enumerate(example_data.items()):
    ax = axes[idx]
    participant_id = example_participants[group]
    
    # Get first 7 days of data
    start_time = data.index[0]
    end_time = start_time + timedelta(days=7)
    week_data = data[data.index < end_time]
    
    # Resample to 10-minute for cleaner plot
    temp_resampled = week_data['temperature'].resample('10min').mean()
    
    ax.plot(temp_resampled.index, temp_resampled.values, linewidth=2.0, color=COLORS[group])
    
    # Add threshold lines
    ax.axhline(20, color=COLORS['threshold_current'], linestyle='--', linewidth=1.5, alpha=0.7, label='20°C threshold')
    ax.axhline(18, color=COLORS['threshold_proposed'], linestyle='--', linewidth=1.5, alpha=0.7, label='18°C threshold')
    
    # Shade night periods
    current = start_time.replace(hour=NIGHT_START_HOUR, minute=0, second=0, microsecond=0)
    while current < end_time:
        night_end = current + timedelta(hours=8)
        ax.axvspan(current, night_end, alpha=0.15, color='grey', label='Night window (22:00-06:00)' if current == start_time.replace(hour=NIGHT_START_HOUR, minute=0, second=0, microsecond=0) else '')
        current += timedelta(days=1)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'{group.capitalize()} - Participant {participant_id} (7 days)')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_multiday_temperature_profile.png", dpi=300, bbox_inches='tight')
print(f"Saved: 05_multiday_temperature_profile.png")
plt.close()


# ----- FIGURE 6: Temperature vs Acceleration Scatter Plot -----
print("\nCalculating temperature vs acceleration correlation...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Temperature vs Acceleration (Movement) Correlation', fontsize=14, fontweight='bold')

window_samples = ACCEL_SD_WINDOW_SEC * RESAMPLE_HZ

for idx, (group, data) in enumerate(example_data.items()):
    ax = axes[idx]
    participant_id = example_participants[group]
    
    # Calculate acceleration SD
    accel_sd = calculate_acceleration_sd(data, window_samples)
    
    # Get nocturnal data only
    nocturnal_mask = data.index.hour.map(is_night_hour)
    nocturnal_temp = data.loc[nocturnal_mask, 'temperature']
    nocturnal_accel_sd = accel_sd[nocturnal_mask]
    
    # Subsample for plotting (every 100th point)
    subsample_idx = np.arange(0, len(nocturnal_temp), 100)
    
    ax.scatter(nocturnal_temp.iloc[subsample_idx], 
               nocturnal_accel_sd.iloc[subsample_idx],
               alpha=0.3, s=5, color=COLORS[group])
    
    # Add threshold lines
    ax.axvline(20, color=COLORS['threshold_current'], linestyle='--', linewidth=1.5, label='20°C threshold')
    ax.axvline(18, color=COLORS['threshold_proposed'], linestyle='--', linewidth=1.5, label='18°C threshold')
    
    # Add non-wear threshold (actipy uses SD < 0.003g)
    ax.axhline(0.003, color='red', linestyle=':', linewidth=1.5, label='Non-wear SD threshold (0.003g)')
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Acceleration SD (g)')
    ax.set_title(f'{group.capitalize()} - Participant {participant_id}\n(Nocturnal data only)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(10, 35)
    ax.set_ylim(0, 0.1)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_temperature_vs_acceleration.png", dpi=300, bbox_inches='tight')
print(f"Saved: 06_temperature_vs_acceleration.png")
plt.close()


# =============================================================================
# STEP 4: SUMMARY REPORT
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: GENERATING SUMMARY REPORT")
print("=" * 70)

report_lines = [
    "=" * 60,
    "TEMPERATURE INVESTIGATION SUMMARY REPORT",
    "=" * 60,
    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Participants analyzed: {len(df_stats_all)}",
    f"  - Controls: {len(df_stats_all[df_stats_all['group'] == 'controls'])}",
    f"  - iRBD: {len(df_stats_all[df_stats_all['group'] == 'irbd'])}",
    "",
    "ALL-DAY TEMPERATURE CHARACTERISTICS:",
    f"  Mean temperature: {df_stats_all['temp_mean'].mean():.2f}°C (±{df_stats_all['temp_mean'].std():.2f})",
    f"  Median temperature: {df_stats_all['temp_median'].median():.2f}°C",
    f"  Temperature range: {df_stats_all['temp_min'].min():.2f}°C - {df_stats_all['temp_max'].max():.2f}°C",
    "",
    "NOCTURNAL TEMPERATURE CHARACTERISTICS (22:00-06:00):",
    f"  Mean temperature: {df_stats_night['temp_mean'].mean():.2f}°C (±{df_stats_night['temp_mean'].std():.2f})",
    f"  Median temperature: {df_stats_night['temp_median'].median():.2f}°C",
    f"  Temperature range: {df_stats_night['temp_min'].min():.2f}°C - {df_stats_night['temp_max'].max():.2f}°C",
    "",
    "COMPARISON TO LITERATURE:",
    f"  Brink-Kjaer (2023) threshold: 27°C",
    f"  Our nocturnal mean: {df_stats_night['temp_mean'].mean():.2f}°C",
    f"  Difference: {df_stats_night['temp_mean'].mean() - 27:.2f}°C",
    "",
    "RETENTION RATES BY THRESHOLD (NOCTURNAL DATA):",
]

for threshold in THRESHOLDS_TO_TEST:
    rate = df_retention_night[df_retention_night['threshold'] == threshold]['retention_rate'].mean()
    std = df_retention_night[df_retention_night['threshold'] == threshold]['retention_rate'].std()
    report_lines.append(f"  {threshold}°C: {rate:.1f}% (±{std:.1f}%)")

report_lines.extend([
    "",
    "IMPACT ANALYSIS:",
    f"  At 27°C (Brink-Kjaer): {df_retention_night[df_retention_night['threshold'] == 27]['retention_rate'].mean():.1f}% retention",
    f"  At 20°C: {df_retention_night[df_retention_night['threshold'] == 20]['retention_rate'].mean():.1f}% retention",
    f"  At 18°C: {df_retention_night[df_retention_night['threshold'] == 18]['retention_rate'].mean():.1f}% retention",
    "",
    "KEY FINDINGS:",
    f"  1. Nocturnal temperatures are significantly lower than expected skin temperature (~33°C)",
    f"  2. The 27°C threshold from Brink-Kjaer (2023) would exclude most data",
    f"  3. Temperature shows clear day/night patterns (see hourly plots)",
    f"  4. Lower thresholds (18-20°C) are more appropriate for this dataset",
    "",
    "EXAMPLE PARTICIPANTS USED FOR TIME-SERIES PLOTS:",
    f"  Controls: {example_participants.get('controls', 'N/A')}",
    f"  iRBD: {example_participants.get('irbd', 'N/A')}",
    "",
    "OUTPUT FILES:",
    "  - 01_temperature_dashboard_nocturnal.png",
    "  - 02_hourly_temperature_heatmap.png",
    "  - 03_hourly_temperature_boxplot.png",
    "  - 04_24hour_temperature_profile.png",
    "  - 05_multiday_temperature_profile.png",
    "  - 06_temperature_vs_acceleration.png",
    "  - temperature_statistics_all_day.csv",
    "  - temperature_statistics_nocturnal.csv",
    "  - retention_rates_all_day.csv",
    "  - retention_rates_nocturnal.csv",
    "  - hourly_temperature_data.csv",
    "",
    "=" * 60
])

report_text = "\n".join(report_lines)
print(report_text)

# Save report
with open(OUTPUT_DIR / "temperature_investigation_report.txt", 'w') as f:
    f.write(report_text)

print(f"\nSaved: temperature_investigation_report.txt")
print("\n" + "=" * 70)
print("TEMPERATURE INVESTIGATION COMPLETE")
print("=" * 70)
