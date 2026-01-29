#!/usr/bin/env python3
"""
Feature Quality Check Script
Validates extracted SSL-Wearables features for v0 and v1 preprocessing versions
"""

import numpy as np
import glob
from pathlib import Path
import sys

def check_feature_file(npz_path):
    """Check a single feature file for quality issues."""
    issues = []
    stats = {}
    
    try:
        data = np.load(npz_path)
        
        # Check required keys
        required_keys = ['features', 'windows_mask', 'participant_id', 'num_nights']
        missing_keys = [k for k in required_keys if k not in data.keys()]
        if missing_keys:
            issues.append(f"Missing keys: {missing_keys}")
            return None, issues
        
        # Extract data
        features = data['features']  # (nights, max_windows, 1024)
        mask = data['windows_mask']  # (nights, max_windows)
        participant_id = str(data['participant_id'])
        num_nights = int(data['num_nights'])
        
        # Check shapes
        if features.ndim != 3:
            issues.append(f"Features has wrong dimensions: {features.ndim} (expected 3)")
        
        if features.shape[2] != 1024:
            issues.append(f"Feature dimension is {features.shape[2]} (expected 1024)")
        
        if features.shape[:2] != mask.shape:
            issues.append(f"Shape mismatch: features {features.shape[:2]} vs mask {mask.shape}")
        
        # Check for NaN or Inf
        if np.any(np.isnan(features)):
            issues.append(f"Contains NaN values")
        
        if np.any(np.isinf(features)):
            issues.append(f"Contains Inf values")
        
        # Check mask validity
        valid_windows = np.sum(mask)
        total_windows = mask.size
        
        if valid_windows == 0:
            issues.append(f"No valid windows (all masked)")
        
        # Calculate statistics
        valid_features = features[mask]
        
        stats = {
            'participant_id': participant_id,
            'num_nights': num_nights,
            'nights_shape': features.shape[0],
            'max_windows': features.shape[1],
            'feature_dim': features.shape[2],
            'total_windows': total_windows,
            'valid_windows': int(valid_windows),
            'masked_windows': int(total_windows - valid_windows),
            'valid_ratio': float(valid_windows / total_windows),
            'feature_mean': float(np.mean(valid_features)),
            'feature_std': float(np.std(valid_features)),
            'feature_min': float(np.min(valid_features)),
            'feature_max': float(np.max(valid_features)),
            'has_issues': len(issues) > 0
        }
        
        return stats, issues
        
    except Exception as e:
        issues.append(f"Error loading file: {str(e)}")
        return None, issues


def check_version(version):
    """Check all feature files for a specific version."""
    print(f"\n{'='*80}")
    print(f"CHECKING FEATURES FOR VERSION: {version}")
    print(f"{'='*80}\n")
    
    # Find all feature files
    feature_dir = Path(f"data/features_{version}")
    control_files = sorted(feature_dir.glob("controls/*.npz"))
    irbd_files = sorted(feature_dir.glob("irbd/*.npz"))
    
    all_files = list(control_files) + list(irbd_files)
    
    if len(all_files) == 0:
        print(f"❌ No feature files found in {feature_dir}")
        return
    
    print(f"Found {len(all_files)} feature files:")
    print(f"  - Controls: {len(control_files)}")
    print(f"  - iRBD: {len(irbd_files)}")
    print()
    
    # Check each file
    all_stats = []
    files_with_issues = []
    
    for i, npz_file in enumerate(all_files, 1):
        stats, issues = check_feature_file(npz_file)
        
        if stats is None:
            print(f"❌ [{i}/{len(all_files)}] {npz_file.name}: FAILED TO LOAD")
            for issue in issues:
                print(f"     - {issue}")
            files_with_issues.append(npz_file.name)
        elif len(issues) > 0:
            print(f"⚠️  [{i}/{len(all_files)}] {npz_file.name}: HAS ISSUES")
            for issue in issues:
                print(f"     - {issue}")
            files_with_issues.append(npz_file.name)
            all_stats.append(stats)
        else:
            all_stats.append(stats)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS FOR {version.upper()}")
    print(f"{'='*80}\n")
    
    if len(all_stats) == 0:
        print("❌ No valid files to analyze")
        return
    
    print(f"✅ Valid files: {len(all_stats)}/{len(all_files)}")
    print(f"❌ Files with issues: {len(files_with_issues)}")
    
    if len(files_with_issues) > 0:
        print(f"\nFiles with issues:")
        for fname in files_with_issues:
            print(f"  - {fname}")
    
    # Aggregate statistics
    total_nights = sum(s['num_nights'] for s in all_stats)
    total_valid_windows = sum(s['valid_windows'] for s in all_stats)
    total_masked_windows = sum(s['masked_windows'] for s in all_stats)
    
    print(f"\n📊 Data Statistics:")
    print(f"  Total participants: {len(all_stats)}")
    print(f"  Total nights: {total_nights}")
    print(f"  Avg nights/participant: {total_nights/len(all_stats):.1f}")
    print(f"  Total valid windows: {total_valid_windows:,}")
    print(f"  Total masked windows: {total_masked_windows:,}")
    print(f"  Valid window ratio: {total_valid_windows/(total_valid_windows+total_masked_windows)*100:.1f}%")
    
    # Feature statistics
    all_means = [s['feature_mean'] for s in all_stats]
    all_stds = [s['feature_std'] for s in all_stats]
    
    print(f"\n📈 Feature Statistics:")
    print(f"  Mean across participants: {np.mean(all_means):.4f} ± {np.std(all_means):.4f}")
    print(f"  Std across participants: {np.mean(all_stds):.4f} ± {np.std(all_stds):.4f}")
    print(f"  Min value: {min(s['feature_min'] for s in all_stats):.4f}")
    print(f"  Max value: {max(s['feature_max'] for s in all_stats):.4f}")
    
    # Check for anomalies
    print(f"\n🔍 Quality Checks:")
    
    # Check for participants with very few nights
    few_nights = [s for s in all_stats if s['num_nights'] < 5]
    if len(few_nights) > 0:
        print(f"  ⚠️  {len(few_nights)} participants with < 5 nights:")
        for s in few_nights[:5]:  # Show first 5
            print(f"     - {s['participant_id']}: {s['num_nights']} nights")
    else:
        print(f"  ✅ All participants have >= 5 nights")
    
    # Check for participants with low valid window ratio
    low_valid = [s for s in all_stats if s['valid_ratio'] < 0.8]
    if len(low_valid) > 0:
        print(f"  ⚠️  {len(low_valid)} participants with < 80% valid windows:")
        for s in low_valid[:5]:  # Show first 5
            print(f"     - {s['participant_id']}: {s['valid_ratio']*100:.1f}% valid")
    else:
        print(f"  ✅ All participants have >= 80% valid windows")
    
    # Check for extreme feature values
    extreme_mean = [s for s in all_stats if abs(s['feature_mean']) > 1.0]
    if len(extreme_mean) > 0:
        print(f"  ⚠️  {len(extreme_mean)} participants with |mean| > 1.0 (unusual for normalized features)")
    else:
        print(f"  ✅ All participants have reasonable feature means")
    
    print()


def main():
    print("="*80)
    print("FEATURE QUALITY CHECK")
    print("="*80)
    
    # Check both versions
    for version in ['v0', 'v1']:
        check_version(version)
    
    print("\n" + "="*80)
    print("QUALITY CHECK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
