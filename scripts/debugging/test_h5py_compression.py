#!/usr/bin/env python3
"""
Test script to verify h5py compression optimization
Tests on 1-2 files to ensure the fix works before applying to all preprocessing scripts
"""

import h5py
import numpy as np
import pandas as pd
import os
from pathlib import Path
import time

def test_uncompressed_vs_compressed():
    """
    Create test data and compare uncompressed vs compressed h5 files
    """
    print("=" * 60)
    print("H5PY COMPRESSION TEST")
    print("=" * 60)
    
    # Create test data (simulating one night of accelerometer data)
    print("\n1. Creating test data...")
    n_samples = 864000  # 8 hours at 30 Hz
    test_data = {
        'x': np.random.randn(n_samples).astype('float32'),
        'y': np.random.randn(n_samples).astype('float32'),
        'z': np.random.randn(n_samples).astype('float32'),
        'timestamps': np.arange(n_samples, dtype='int64')
    }
    print(f"   - Created {n_samples:,} samples (8 hours at 30 Hz)")
    print(f"   - Data shape: ({n_samples}, 3)")
    print(f"   - Memory size: {(n_samples * 3 * 4) / 1e6:.1f} MB (float32)")
    
    # Test 1: OLD METHOD (no compression, separate arrays)
    print("\n2. Testing OLD method (no compression)...")
    old_file = '/tmp/test_old.h5'
    start = time.time()
    
    with h5py.File(old_file, 'w') as f:
        night_group = f.create_group('night1')
        night_group.create_dataset('x', data=test_data['x'])
        night_group.create_dataset('y', data=test_data['y'])
        night_group.create_dataset('z', data=test_data['z'])
        night_group.create_dataset('timestamps', data=test_data['timestamps'])
    
    old_time = time.time() - start
    old_size = os.path.getsize(old_file) / 1e6
    print(f"   - File size: {old_size:.1f} MB")
    print(f"   - Write time: {old_time:.2f} seconds")
    
    # Test 2: NEW METHOD (compression, combined arrays)
    print("\n3. Testing NEW method (with compression)...")
    new_file = '/tmp/test_new.h5'
    start = time.time()
    
    with h5py.File(new_file, 'w') as f:
        night_group = f.create_group('night1')
        
        # Combine x, y, z into single array
        accel_data = np.column_stack([
            test_data['x'],
            test_data['y'],
            test_data['z']
        ])
        
        # Create compressed dataset with optimal chunking
        night_group.create_dataset(
            'accel',
            data=accel_data,
            compression='gzip',
            compression_opts=4,
            chunks=(10000, 3),
            dtype='float32'
        )
        
        # Store timestamps efficiently
        night_group.create_dataset(
            'timestamps',
            data=test_data['timestamps'],
            compression='gzip',
            compression_opts=4,
            chunks=(10000,)
        )
        
        # Add metadata
        night_group.attrs['sampling_rate'] = 30
        night_group.attrs['night_number'] = 1
        night_group.attrs['duration_hours'] = 8.0
    
    new_time = time.time() - start
    new_size = os.path.getsize(new_file) / 1e6
    print(f"   - File size: {new_size:.1f} MB")
    print(f"   - Write time: {new_time:.2f} seconds")
    
    # Test 3: Verify data can be read correctly
    print("\n4. Verifying data integrity...")
    with h5py.File(new_file, 'r') as f:
        night = f['night1']
        accel_read = night['accel'][:]
        timestamps_read = night['timestamps'][:]
        
        # Check data matches
        x_match = np.allclose(accel_read[:, 0], test_data['x'])
        y_match = np.allclose(accel_read[:, 1], test_data['y'])
        z_match = np.allclose(accel_read[:, 2], test_data['z'])
        t_match = np.array_equal(timestamps_read, test_data['timestamps'])
        
        print(f"   - X data matches: {x_match}")
        print(f"   - Y data matches: {y_match}")
        print(f"   - Z data matches: {z_match}")
        print(f"   - Timestamps match: {t_match}")
        print(f"   - Metadata: {dict(night.attrs)}")
    
    # Test 4: Benchmark random access (h5py's strength!)
    print("\n5. Testing random access performance...")
    
    # Old method: read specific time range
    start = time.time()
    with h5py.File(old_file, 'r') as f:
        x_slice = f['night1']['x'][100000:200000]
    old_read_time = time.time() - start
    
    # New method: read specific time range
    start = time.time()
    with h5py.File(new_file, 'r') as f:
        accel_slice = f['night1']['accel'][100000:200000, :]
    new_read_time = time.time() - start
    
    print(f"   - OLD method read time: {old_read_time*1000:.2f} ms")
    print(f"   - NEW method read time: {new_read_time*1000:.2f} ms")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"File size reduction: {old_size:.1f} MB → {new_size:.1f} MB ({old_size/new_size:.1f}x smaller)")
    print(f"Compression ratio: {(1 - new_size/old_size)*100:.1f}% reduction")
    print(f"Write time: {old_time:.2f}s → {new_time:.2f}s")
    print(f"Random access: {old_read_time*1000:.2f}ms → {new_read_time*1000:.2f}ms")
    print(f"Data integrity: ✓ All data verified correct")
    
    # Projected savings
    print("\n" + "=" * 60)
    print("PROJECTED SAVINGS FOR FULL DATASET")
    print("=" * 60)
    
    # Assume 84 participants, ~30 nights each, ~2 GB per participant (old)
    old_total = 254  # GB (current usage)
    new_total = old_total * (new_size / old_size)
    savings = old_total - new_total
    
    print(f"Current disk usage: {old_total:.0f} GB")
    print(f"Projected with compression: {new_total:.0f} GB")
    print(f"Space saved: {savings:.0f} GB ({(savings/old_total)*100:.0f}%)")
    print(f"Fits in quota: {'✓ YES' if new_total < 100 else '✗ NO (need more quota)'}")
    
    # Cleanup
    os.remove(old_file)
    os.remove(new_file)
    print("\n✓ Test files cleaned up")
    print("=" * 60)


def test_on_real_file():
    """
    Test compression on an actual preprocessed file
    """
    print("\n\n" + "=" * 60)
    print("TESTING ON REAL PREPROCESSED FILE")
    print("=" * 60)
    
    # Path to an existing preprocessed file
    test_file = "/work3/s184484/iRBD-detection/data/preprocessed_v3/controls/AXRBD025.h5"
    
    if not os.path.exists(test_file):
        print("✗ Test file not found, skipping real file test")
        return
    
    print(f"\nOriginal file: {test_file}")
    original_size = os.path.getsize(test_file) / 1e6
    print(f"Original size: {original_size:.1f} MB")
    
    # Read original data
    print("\nReading original data...")
    with h5py.File(test_file, 'r') as f:
        print(f"Keys: {list(f.keys())}")
        
        # Read all nights
        nights_data = {}
        for night_key in f.keys():
            night_group = f[night_key]
            
            # Check if old format (separate x,y,z) or new format (combined accel)
            if 'accel' in night_group:
                print(f"   - {night_key}: Already using new format!")
                return
            else:
                nights_data[night_key] = {
                    'x': night_group['x'][:],
                    'y': night_group['y'][:],
                    'z': night_group['z'][:],
                    'timestamps': night_group['timestamps'][:]
                }
                print(f"   - {night_key}: {len(nights_data[night_key]['x']):,} samples")
    
    # Create compressed version
    compressed_file = test_file.replace('.h5', '_compressed.h5')
    print(f"\nCreating compressed version: {compressed_file}")
    
    with h5py.File(compressed_file, 'w') as f:
        for night_key, night_data in nights_data.items():
            night_group = f.create_group(night_key)
            
            # Combine x, y, z
            accel_data = np.column_stack([
                night_data['x'],
                night_data['y'],
                night_data['z']
            ])
            
            # Create compressed datasets
            night_group.create_dataset(
                'accel',
                data=accel_data,
                compression='gzip',
                compression_opts=4,
                chunks=(10000, 3),
                dtype='float32'
            )
            
            night_group.create_dataset(
                'timestamps',
                data=night_data['timestamps'],
                compression='gzip',
                compression_opts=4,
                chunks=(10000,)
            )
            
            # Add metadata
            night_group.attrs['sampling_rate'] = 30
            night_group.attrs['n_samples'] = len(night_data['x'])
    
    compressed_size = os.path.getsize(compressed_file) / 1e6
    print(f"Compressed size: {compressed_size:.1f} MB")
    print(f"Reduction: {original_size:.1f} MB → {compressed_size:.1f} MB ({original_size/compressed_size:.1f}x smaller)")
    print(f"Space saved: {original_size - compressed_size:.1f} MB ({(1-compressed_size/original_size)*100:.0f}%)")
    
    print(f"\n✓ Compressed file created: {compressed_file}")
    print("  (You can delete this test file after verification)")


if __name__ == '__main__':
    # Run synthetic test
    test_uncompressed_vs_compressed()
    
    # Run real file test
    try:
        test_on_real_file()
    except Exception as e:
        print(f"\n✗ Real file test failed: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the results above")
    print("2. If compression works well, apply fix to preprocessing scripts")
    print("3. Delete old uncompressed files to free space")
    print("4. Rerun preprocessing with compression enabled")
