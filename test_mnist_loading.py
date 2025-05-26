#!/usr/bin/env python3
"""
Test script to verify MNIST dataset loading from raw files.
"""

import sys
import os
sys.path.append('src')

from experiment_design.datasets.mnist import load_mnist_dataset
from pathlib import Path

def test_mnist_loading():
    """Test loading MNIST dataset from raw files."""
    print("Testing MNIST dataset loading...")
    
    # Test training set
    try:
        train_dataset = load_mnist_dataset(
            root="data/MNIST",
            train=True,
            max_samples=10  # Just load a few samples for testing
        )
        print(f"✓ Training dataset loaded successfully: {len(train_dataset)} samples")
        
        # Test getting a sample
        sample = train_dataset[0]
        image, label, filename = sample
        print(f"✓ Sample 0: label={label}, filename={filename}, image shape={image.shape}")
        
    except Exception as e:
        print(f"✗ Error loading training dataset: {e}")
        return False
    
    # Test test set
    try:
        test_dataset = load_mnist_dataset(
            root="data/MNIST",
            train=False,
            max_samples=10  # Just load a few samples for testing
        )
        print(f"✓ Test dataset loaded successfully: {len(test_dataset)} samples")
        
        # Test getting a sample
        sample = test_dataset[0]
        image, label, filename = sample
        print(f"✓ Sample 0: label={label}, filename={filename}, image shape={image.shape}")
        
    except Exception as e:
        print(f"✗ Error loading test dataset: {e}")
        return False
    
    print("✓ All MNIST dataset tests passed!")
    return True

if __name__ == "__main__":
    success = test_mnist_loading()
    sys.exit(0 if success else 1) 