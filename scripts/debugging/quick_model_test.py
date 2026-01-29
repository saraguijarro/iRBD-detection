#!/usr/bin/env python3
"""
Quick test script to verify SSL-Wearables model feature extraction
Run this BEFORE deploying the full pipeline to confirm dimensions
"""

import torch
import numpy as np

print("="*80)
print("SSL-WEARABLES MODEL TEST")
print("="*80)

# Load model
print("\n1. Loading model...")
model = torch.hub.load('OxWearables/ssl-wearables', 'harnet10', pretrained=True, trust_repo=True)
model.eval()

print(f"   Model type: {type(model)}")
print(f"   Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")

# Test input
test_input = torch.randn(2, 3, 300)
print(f"\n2. Test input shape: {test_input.shape}")

# Test different extraction methods
print('\n3. Testing full model output:')
with torch.no_grad():
    output = model(test_input)
    print(f'   Shape: {output.shape}')
    print(f'   This is the CLASSIFIER output (wrong for feature extraction!)')

print('\n4. Testing feature extractor output:')
if hasattr(model, 'feature_extractor'):
    with torch.no_grad():
        features = model.feature_extractor(test_input)
        print(f'   Raw shape: {features.shape}')
        flattened = features.view(2, -1)
        print(f'   Flattened shape: {flattened.shape}')
        print(f'   Feature dimension: {flattened.shape[-1]}')
        print(f'   This is the CORRECT output for feature extraction!')
else:
    print('   ERROR: No feature_extractor attribute')

print('\n5. Testing for alternative attributes:')
if hasattr(model, 'encoder'):
    print('   Found: encoder')
else:
    print('   Not found: encoder')
    
if hasattr(model, 'features'):
    print('   Found: features')
else:
    print('   Not found: features')

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\nIf you see 'Feature dimension: 512' or '1024', the model is working correctly!")
print("Use model.feature_extractor(x) in your pipeline, NOT model(x)")
