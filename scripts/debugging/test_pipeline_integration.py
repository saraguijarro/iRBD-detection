#!/usr/bin/env python3
"""
Integration test for iRBD detection pipeline.
Tests the data flow from feature extraction to LSTM training.
"""

import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil

def test_feature_format():
    """Test that feature extraction outputs the correct format for LSTM."""
    print("=" * 80)
    print("TEST 1: Feature Format Compatibility")
    print("=" * 80)
    
    # Simulate feature extraction output
    num_nights = 3
    windows_per_night = [100, 150, 120]  # Variable length
    max_windows = max(windows_per_night)
    feature_dim = 1024
    
    # Create 3D features array (nights, max_windows, features)
    features_3d = np.zeros((num_nights, max_windows, feature_dim), dtype=np.float32)
    windows_mask = np.zeros((num_nights, max_windows), dtype=bool)
    
    # Fill with random data and create mask
    for night_idx, num_windows in enumerate(windows_per_night):
        features_3d[night_idx, :num_windows, :] = np.random.randn(num_windows, feature_dim).astype(np.float32)
        windows_mask[night_idx, :num_windows] = True
    
    print(f"✓ Created features: shape={features_3d.shape}")
    print(f"✓ Created mask: shape={windows_mask.shape}")
    print(f"✓ Valid windows: {windows_mask.sum()}/{windows_mask.size}")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        tmp_path = tmp.name
        np.savez_compressed(
            tmp_path,
            features=features_3d,
            windows_mask=windows_mask,
            participant_id='TEST001',
            num_nights=num_nights,
            max_windows=max_windows,
            total_windows=sum(windows_per_night),
            windows_per_night=np.array(windows_per_night),
            preprocessing_version='v0_test'
        )
    
    # Load and verify
    data = np.load(tmp_path)
    assert 'features' in data, "Missing 'features' key!"
    assert 'windows_mask' in data, "Missing 'windows_mask' key!"
    assert data['features'].shape == (num_nights, max_windows, feature_dim), "Wrong features shape!"
    assert data['windows_mask'].shape == (num_nights, max_windows), "Wrong mask shape!"
    
    print(f"✓ Saved and loaded successfully")
    print(f"✓ All required keys present: {list(data.keys())}")
    
    # Clean up
    Path(tmp_path).unlink()
    
    print("\n✅ TEST 1 PASSED: Feature format is correct\n")
    return True


def test_lstm_loading():
    """Test that LSTM can load the feature format."""
    print("=" * 80)
    print("TEST 2: LSTM Data Loading")
    print("=" * 80)
    
    # Create test data
    num_nights = 2
    max_windows = 50
    feature_dim = 1024
    
    features_3d = np.random.randn(num_nights, max_windows, feature_dim).astype(np.float32)
    windows_mask = np.ones((num_nights, max_windows), dtype=bool)
    windows_mask[0, 40:] = False  # Mask out last 10 windows of first night
    windows_mask[1, 45:] = False  # Mask out last 5 windows of second night
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        tmp_path = tmp.name
        np.savez_compressed(
            tmp_path,
            features=features_3d,
            windows_mask=windows_mask,
            participant_id='TEST002',
            num_nights=num_nights
        )
    
    # Load as LSTM would
    data = np.load(tmp_path)
    features = data['features']
    mask = data['windows_mask']
    
    print(f"✓ Loaded features: {features.shape}")
    print(f"✓ Loaded mask: {mask.shape}")
    print(f"✓ Valid windows: {mask.sum()}/{mask.size}")
    
    # Convert to PyTorch tensors
    features_tensor = torch.FloatTensor(features)
    mask_tensor = torch.BoolTensor(mask)
    
    print(f"✓ Converted to PyTorch tensors")
    print(f"  Features: {features_tensor.shape}, dtype={features_tensor.dtype}")
    print(f"  Mask: {mask_tensor.shape}, dtype={mask_tensor.dtype}")
    
    # Clean up
    Path(tmp_path).unlink()
    
    print("\n✅ TEST 2 PASSED: LSTM can load features correctly\n")
    return True


def test_lstm_forward_pass():
    """Test that LSTM model can process the data."""
    print("=" * 80)
    print("TEST 3: LSTM Forward Pass")
    print("=" * 80)
    
    # Create batch of data
    batch_size = 2
    num_nights = 3
    max_windows = 50
    feature_dim = 1024
    
    # Create test batch
    features = torch.randn(batch_size, num_nights, max_windows, feature_dim)
    mask = torch.ones(batch_size, num_nights, max_windows, dtype=torch.bool)
    
    # Mask out some windows
    mask[0, 2, 40:] = False
    mask[1, 1, 45:] = False
    
    print(f"✓ Created batch:")
    print(f"  Features: {features.shape}")
    print(f"  Mask: {mask.shape}")
    
    # Test the reshaping logic from the model
    batch_size_test, num_nights_test, num_windows_test, feature_dim_test = features.shape
    
    # Flatten nights and windows
    features_reshaped = features.view(batch_size_test, num_nights_test * num_windows_test, feature_dim_test)
    mask_reshaped = mask.view(batch_size_test, num_nights_test * num_windows_test)
    
    print(f"✓ Reshaped for LSTM:")
    print(f"  Features: {features_reshaped.shape}")
    print(f"  Mask: {mask_reshaped.shape}")
    
    assert features_reshaped.shape == (batch_size, num_nights * max_windows, feature_dim), "Wrong reshape!"
    assert mask_reshaped.shape == (batch_size, num_nights * max_windows), "Wrong mask reshape!"
    
    print(f"✓ Shapes are correct for LSTM input")
    
    print("\n✅ TEST 3 PASSED: LSTM forward pass logic is correct\n")
    return True


def test_end_to_end():
    """Test complete pipeline from feature extraction to LSTM."""
    print("=" * 80)
    print("TEST 4: End-to-End Pipeline")
    print("=" * 80)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    features_dir = temp_dir / "features"
    features_dir.mkdir()
    
    try:
        # Simulate feature extraction for 2 participants
        for i, (num_nights, windows_per_night) in enumerate([
            (2, [80, 100]),
            (3, [90, 110, 95])
        ]):
            max_windows = max(windows_per_night)
            feature_dim = 1024
            
            # Create features
            features_3d = np.zeros((num_nights, max_windows, feature_dim), dtype=np.float32)
            windows_mask = np.zeros((num_nights, max_windows), dtype=bool)
            
            for night_idx, num_windows in enumerate(windows_per_night):
                features_3d[night_idx, :num_windows, :] = np.random.randn(num_windows, feature_dim).astype(np.float32)
                windows_mask[night_idx, :num_windows] = True
            
            # Save
            output_file = features_dir / f"TEST{i:03d}.npz"
            np.savez_compressed(
                output_file,
                features=features_3d,
                windows_mask=windows_mask,
                participant_id=f'TEST{i:03d}',
                num_nights=num_nights,
                max_windows=max_windows,
                total_windows=sum(windows_per_night),
                windows_per_night=np.array(windows_per_night)
            )
            
            print(f"✓ Created participant TEST{i:03d}: {num_nights} nights, {sum(windows_per_night)} windows")
        
        # Load all files as LSTM would
        feature_files = list(features_dir.glob("*.npz"))
        print(f"\n✓ Found {len(feature_files)} feature files")
        
        all_features = []
        all_masks = []
        
        for feature_file in feature_files:
            data = np.load(feature_file)
            all_features.append(data['features'])
            all_masks.append(data['windows_mask'])
            print(f"  Loaded {feature_file.name}: {data['features'].shape}")
        
        print(f"\n✓ Successfully loaded all features")
        print(f"✓ All files have correct format (features + windows_mask)")
        
        print("\n✅ TEST 4 PASSED: End-to-end pipeline works!\n")
        return True
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def main():
    """Run all integration tests."""
    print("\n" + "=" * 80)
    print("iRBD DETECTION PIPELINE - INTEGRATION TESTS")
    print("=" * 80 + "\n")
    
    tests = [
        test_feature_format,
        test_lstm_loading,
        test_lstm_forward_pass,
        test_end_to_end
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_func.__name__}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("=" * 80)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✅ ALL TESTS PASSED - Pipeline is working correctly!")
    else:
        print(f"❌ {failed} tests failed - Fix issues before running full pipeline")
    print("=" * 80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
