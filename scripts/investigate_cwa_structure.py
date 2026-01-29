#!/usr/bin/env python3
"""
investigate_cwa_structure.py

Investigates the structure and metadata of Axivity .cwa files
to verify device specifications and data characteristics.

Usage:
    python investigate_cwa_structure.py /path/to/cwa/files/ --output results.json
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
from collections import defaultdict
from datetime import datetime
import argparse

def investigate_cwa_with_actipy(cwa_path):
    """
    Extract comprehensive metadata from a .cwa file using actipy.
    
    Parameters:
    -----------
    cwa_path : str
        Path to the .cwa file
        
    Returns:
    --------
    dict : Dictionary containing file metadata and characteristics
    """
    try:
        import actipy
        
        # Read the .cwa file
        print(f"  Reading file with actipy...")
        data, info = actipy.read_device(str(cwa_path), 
                                         lowpass_hz=20,
                                         calibrate_gravity=True,
                                         detect_nonwear=False)
        
        # Calculate duration
        start_time = data.index[0]
        end_time = data.index[-1]
        duration_seconds = (end_time - start_time).total_seconds()
        duration_hours = duration_seconds / 3600
        duration_days = duration_hours / 24
        
        # Infer sampling rate from data
        time_diffs = pd.Series(data.index).diff().dt.total_seconds()
        inferred_sample_rate = 1.0 / time_diffs.median()
        
        # Check for temperature data
        has_temperature = 'temperature' in data.columns
        
        # Temperature statistics if available
        temp_stats = {}
        if has_temperature:
            temp_data = data['temperature'].dropna()
            if len(temp_data) > 0:
                temp_stats = {
                    'mean': float(temp_data.mean()),
                    'std': float(temp_data.std()),
                    'min': float(temp_data.min()),
                    'max': float(temp_data.max()),
                    'samples': len(temp_data)
                }
        
        # Accelerometer statistics
        accel_stats = {}
        for axis in ['x', 'y', 'z']:
            if axis in data.columns:
                accel_stats[axis] = {
                    'mean': float(data[axis].mean()),
                    'std': float(data[axis].std()),
                    'min': float(data[axis].min()),
                    'max': float(data[axis].max())
                }
        
        # Build metadata dictionary
        metadata = {
            'file_path': str(cwa_path),
            'file_name': cwa_path.name,
            'file_size_mb': os.path.getsize(cwa_path) / (1024 * 1024),
            
            # Device information from info dict
            'device_id': info.get('DeviceId', 'Unknown'),
            'device_type': info.get('DeviceType', 'Unknown'),
            'hardware_version': info.get('HardwareVersion', 'Unknown'),
            'firmware_version': info.get('FirmwareVersion', 'Unknown'),
            
            # Sampling information
            'sample_rate_hz': info.get('SampleRate', inferred_sample_rate),
            'inferred_sample_rate_hz': float(inferred_sample_rate),
            'num_samples': len(data),
            
            # Time information
            'start_time': str(start_time),
            'end_time': str(end_time),
            'duration_hours': float(duration_hours),
            'duration_days': float(duration_days),
            
            # Sensor information
            'has_accelerometer': all(axis in data.columns for axis in ['x', 'y', 'z']),
            'has_temperature': has_temperature,
            'data_columns': list(data.columns),
            
            # Accelerometer range (from info or inferred from data)
            'accel_range_g': info.get('AccelRange', 'Unknown'),
            
            # Statistics
            'temperature_stats': temp_stats,
            'accelerometer_stats': accel_stats,
            
            # Additional info
            'calibration_info': {
                'x_offset': info.get('CalibrationX', {}).get('offset', 'Unknown'),
                'y_offset': info.get('CalibrationY', {}).get('offset', 'Unknown'),
                'z_offset': info.get('CalibrationZ', {}).get('offset', 'Unknown'),
            } if 'CalibrationX' in info else 'Not available'
        }
        
        return metadata
        
    except ImportError:
        return {
            'file_path': str(cwa_path),
            'error': 'actipy not installed. Install with: pip install actipy'
        }
    except Exception as e:
        return {
            'file_path': str(cwa_path),
            'file_name': cwa_path.name,
            'error': str(e)
        }

def investigate_all_cwa_files(data_dir, output_file='cwa_investigation_results.json', max_files=None):
    """
    Investigate all .cwa files in a directory and save results.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing .cwa files
    output_file : str
        Output JSON file path
    max_files : int or None
        Maximum number of files to process (for testing)
    """
    data_path = Path(data_dir)
    
    # Find all .cwa files (both lowercase and uppercase)
    print(f"Searching for .cwa/.CWA files in {data_dir}...")
    cwa_files_lower = list(data_path.rglob('*.cwa'))
    cwa_files_upper = list(data_path.rglob('*.CWA'))
    cwa_files = cwa_files_lower + cwa_files_upper
    
    if len(cwa_files) == 0:
        print(f"ERROR: No .cwa/.CWA files found in {data_dir}")
        return None
    
    print(f"Found {len(cwa_files)} .cwa files")
    
    if max_files:
        cwa_files = cwa_files[:max_files]
        print(f"Processing first {max_files} files only (for testing)")
    
    results = []
    summary_stats = defaultdict(list)
    
    for i, cwa_file in enumerate(cwa_files, 1):
        print(f"\n[{i}/{len(cwa_files)}] Processing: {cwa_file.name}")
        
        metadata = investigate_cwa_with_actipy(cwa_file)
        results.append(metadata)
        
        # Collect statistics (only from successful reads)
        if 'error' not in metadata:
            summary_stats['sample_rates'].append(metadata.get('sample_rate_hz', None))
            summary_stats['inferred_sample_rates'].append(metadata.get('inferred_sample_rate_hz', None))
            summary_stats['durations_hours'].append(metadata.get('duration_hours', None))
            summary_stats['durations_days'].append(metadata.get('duration_days', None))
            summary_stats['file_sizes_mb'].append(metadata.get('file_size_mb', None))
            summary_stats['has_temperature'].append(metadata.get('has_temperature', False))
            summary_stats['device_ids'].append(metadata.get('device_id', None))
            
            accel_range = metadata.get('accel_range_g', 'Unknown')
            if accel_range != 'Unknown':
                summary_stats['accel_ranges'].append(accel_range)
    
    # Calculate summary statistics
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    summary = {
        'total_files': len(cwa_files),
        'successful_reads': len(successful_results),
        'failed_reads': len(failed_results),
        'failed_files': [r['file_name'] for r in failed_results] if failed_results else [],
    }
    
    # Sample rate statistics
    if summary_stats['sample_rates']:
        unique_rates = list(set([r for r in summary_stats['sample_rates'] if r is not None]))
        summary['sample_rate_hz'] = {
            'unique_values': unique_rates,
            'most_common': max(set(summary_stats['sample_rates']), key=summary_stats['sample_rates'].count)
        }
    
    # Inferred sample rate statistics
    if summary_stats['inferred_sample_rates']:
        inferred_rates = [r for r in summary_stats['inferred_sample_rates'] if r is not None]
        summary['inferred_sample_rate_hz'] = {
            'mean': float(np.mean(inferred_rates)),
            'std': float(np.std(inferred_rates)),
            'min': float(np.min(inferred_rates)),
            'max': float(np.max(inferred_rates))
        }
    
    # Duration statistics
    if summary_stats['durations_hours']:
        durations_hours = [d for d in summary_stats['durations_hours'] if d is not None]
        durations_days = [d for d in summary_stats['durations_days'] if d is not None]
        summary['duration'] = {
            'hours': {
                'mean': float(np.mean(durations_hours)),
                'std': float(np.std(durations_hours)),
                'min': float(np.min(durations_hours)),
                'max': float(np.max(durations_hours))
            },
            'days': {
                'mean': float(np.mean(durations_days)),
                'std': float(np.std(durations_days)),
                'min': float(np.min(durations_days)),
                'max': float(np.max(durations_days))
            }
        }
    
    # File size statistics
    if summary_stats['file_sizes_mb']:
        file_sizes = [s for s in summary_stats['file_sizes_mb'] if s is not None]
        summary['file_size_mb'] = {
            'mean': float(np.mean(file_sizes)),
            'std': float(np.std(file_sizes)),
            'min': float(np.min(file_sizes)),
            'max': float(np.max(file_sizes))
        }
    
    # Temperature sensor statistics
    if summary_stats['has_temperature']:
        summary['temperature_sensor'] = {
            'files_with_temp': sum(summary_stats['has_temperature']),
            'percentage': 100 * sum(summary_stats['has_temperature']) / len(summary_stats['has_temperature'])
        }
    
    # Accelerometer range
    if summary_stats['accel_ranges']:
        summary['accelerometer_range_g'] = {
            'unique_values': list(set(summary_stats['accel_ranges'])),
            'most_common': max(set(summary_stats['accel_ranges']), key=summary_stats['accel_ranges'].count)
        }
    
    # Device IDs
    if summary_stats['device_ids']:
        unique_devices = list(set([d for d in summary_stats['device_ids'] if d != 'Unknown']))
        summary['devices'] = {
            'unique_device_ids': unique_devices,
            'num_unique_devices': len(unique_devices)
        }
    
    # Save results
    output = {
        'investigation_date': str(datetime.now()),
        'data_directory': str(data_dir),
        'summary': summary,
        'individual_files': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*70}")
    print("INVESTIGATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total files found: {summary['total_files']}")
    print(f"Successfully read: {summary['successful_reads']}")
    print(f"Failed to read: {summary['failed_reads']}")
    
    if summary['failed_reads'] > 0:
        print(f"\nFailed files:")
        for fname in summary['failed_files']:
            print(f"  - {fname}")
    
    if 'sample_rate_hz' in summary:
        print(f"\nSampling rate: {summary['sample_rate_hz']['most_common']} Hz")
        if len(summary['sample_rate_hz']['unique_values']) > 1:
            print(f"  (Multiple rates found: {summary['sample_rate_hz']['unique_values']})")
    
    if 'inferred_sample_rate_hz' in summary:
        print(f"Inferred sample rate: {summary['inferred_sample_rate_hz']['mean']:.2f} ± {summary['inferred_sample_rate_hz']['std']:.2f} Hz")
    
    if 'accelerometer_range_g' in summary:
        print(f"Accelerometer range: ±{summary['accelerometer_range_g']['most_common']}g")
    
    if 'duration' in summary:
        print(f"\nRecording duration:")
        print(f"  Hours: {summary['duration']['hours']['mean']:.1f} ± {summary['duration']['hours']['std']:.1f}")
        print(f"         (min: {summary['duration']['hours']['min']:.1f}, max: {summary['duration']['hours']['max']:.1f})")
        print(f"  Days:  {summary['duration']['days']['mean']:.1f} ± {summary['duration']['days']['std']:.1f}")
        print(f"         (min: {summary['duration']['days']['min']:.1f}, max: {summary['duration']['days']['max']:.1f})")
    
    if 'temperature_sensor' in summary:
        print(f"\nTemperature sensor:")
        print(f"  Files with temperature: {summary['temperature_sensor']['files_with_temp']} ({summary['temperature_sensor']['percentage']:.1f}%)")
    
    if 'devices' in summary:
        print(f"\nUnique devices: {summary['devices']['num_unique_devices']}")
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    return output

def main():
    parser = argparse.ArgumentParser(
        description='Investigate Axivity .cwa file structure and metadata',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python investigate_cwa_structure.py /path/to/cwa/files/
  python investigate_cwa_structure.py /path/to/cwa/files/ --output my_results.json
  python investigate_cwa_structure.py /path/to/cwa/files/ --max-files 5
        """
    )
    
    parser.add_argument('data_dir', type=str, 
                       help='Directory containing .cwa files (will search recursively)')
    parser.add_argument('--output', type=str, default='/work3/s184484/iRBD-detection/results/cwa_investigation_results.json',
                       help='Output JSON file (default: cwa_investigation_results.json)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Run investigation
    investigate_all_cwa_files(args.data_dir, args.output, args.max_files)

if __name__ == "__main__":
    main()
