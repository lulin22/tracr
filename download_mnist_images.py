#!/usr/bin/env python3
"""
Script to download and extract MNIST images into a proper directory structure.
This will create train/test directories with class subdirectories (0-9).
"""

import os
import gzip
import struct
import numpy as np
from PIL import Image
from pathlib import Path
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_mnist_files():
    """Download MNIST files if they don't exist."""
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz", 
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    raw_dir = Path("data/MNIST/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        file_path = raw_dir / file
        if not file_path.exists():
            logger.info(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, file_path)
        else:
            logger.info(f"{file} already exists")

def read_mnist_images(file_path):
    """Read MNIST image file."""
    with gzip.open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}")
        
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(num_images, rows, cols)

def read_mnist_labels(file_path):
    """Read MNIST label file."""
    with gzip.open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number: {magic}")
        
        return np.frombuffer(f.read(), dtype=np.uint8)

def extract_mnist_images():
    """Extract MNIST images and organize them into train/test directories."""
    raw_dir = Path("data/MNIST/raw")
    processed_dir = Path("data/MNIST/processed")
    
    # Create directory structure
    for split in ["train", "test"]:
        for class_idx in range(10):
            class_dir = processed_dir / split / str(class_idx)
            class_dir.mkdir(parents=True, exist_ok=True)
    
    # Process training data
    logger.info("Processing training data...")
    train_images = read_mnist_images(raw_dir / "train-images-idx3-ubyte.gz")
    train_labels = read_mnist_labels(raw_dir / "train-labels-idx1-ubyte.gz")
    
    for idx, (image, label) in enumerate(zip(train_images, train_labels)):
        img = Image.fromarray(image, mode='L')
        img_path = processed_dir / "train" / str(label) / f"{idx:05d}.png"
        img.save(img_path)
        
        if idx % 10000 == 0:
            logger.info(f"Processed {idx} training images")
    
    logger.info(f"Finished processing {len(train_images)} training images")
    
    # Process test data
    logger.info("Processing test data...")
    test_images = read_mnist_images(raw_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = read_mnist_labels(raw_dir / "t10k-labels-idx1-ubyte.gz")
    
    for idx, (image, label) in enumerate(zip(test_images, test_labels)):
        img = Image.fromarray(image, mode='L')
        img_path = processed_dir / "test" / str(label) / f"{idx:05d}.png"
        img.save(img_path)
        
        if idx % 1000 == 0:
            logger.info(f"Processed {idx} test images")
    
    logger.info(f"Finished processing {len(test_images)} test images")

def main():
    """Main function to download and extract MNIST images."""
    logger.info("Starting MNIST image extraction...")
    
    # Download files if needed
    download_mnist_files()
    
    # Extract images
    extract_mnist_images()
    
    logger.info("MNIST image extraction completed!")
    logger.info("Images are organized in data/MNIST/processed/train/ and data/MNIST/processed/test/")
    logger.info("Each subdirectory (0-9) contains images for that digit class")

if __name__ == "__main__":
    main() 