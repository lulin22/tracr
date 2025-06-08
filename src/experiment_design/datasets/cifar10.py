"""CIFAR-10 dataset implementation"""

import logging
import pickle
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from .core import (
    BaseDataset,
    DatasetPathError,
    DatasetProcessingError,
    DatasetTransformError,
    TransformFactory,
    TransformType,
)

logger = logging.getLogger("split_computing_logger")


class CIFAR10Dataset(BaseDataset):
    """Dataset implementation for loading and processing CIFAR-10 data."""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_samples: int = -1,
    ) -> None:
        """Initialize CIFAR-10 dataset with specified parameters.
        
        Args:
            root: Root directory containing CIFAR-10 data
            train: Whether to load training or test data
            transform: Transform to apply to images
            target_transform: Transform to apply to labels
            max_samples: Maximum number of samples to load (-1 for all)
        """
        super().__init__(root, transform, target_transform, max_samples)
        self.train = train
        self._initialize_paths(root)
        if not self.transform:
            # CIFAR-10 uses 32x32 RGB images, create appropriate transform
            # Use the EXACT same normalization as in the training script cifar.py:
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # This maps pixel values from [0,1] to [-1,1] range
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self._initialize_dataset(max_samples)

    def _initialize_paths(self, root: Optional[Union[str, Path]]) -> None:
        """Set up dataset paths and verify existence."""
        if root is None:
            # Default to data/CIFAR/cifar-10-batches-py (updated path)
            self.root = self.data_source_dir / "CIFAR" / "cifar-10-batches-py"
        else:
            root_path = Path(root)
            # Check if root points directly to cifar-10-batches-py or if we need to append it
            if root_path.name == "cifar-10-batches-py":
                self.root = root_path
            else:
                # Assume root points to the parent directory (e.g., "data/CIFAR")
                # and append the cifar-10-batches-py subdirectory
                self.root = root_path / "cifar-10-batches-py"
        
        if not self.root.exists():
            logger.error(f"CIFAR-10 root directory not found: {self.root}")
            raise DatasetPathError("CIFAR-10 root directory not found", path=str(self.root))
        
        # Define the expected files
        self.meta_file = self.root / "batches.meta"
        if self.train:
            self.data_files = [self.root / f"data_batch_{i}" for i in range(1, 6)]
        else:
            self.data_files = [self.root / "test_batch"]
        
        # Verify files exist
        if not self.meta_file.exists():
            logger.error(f"CIFAR-10 metadata file not found: {self.meta_file}")
            raise DatasetPathError("CIFAR-10 metadata file not found", path=str(self.meta_file))
        
        for data_file in self.data_files:
            if not data_file.exists():
                logger.error(f"CIFAR-10 data file not found: {data_file}")
                raise DatasetPathError(f"CIFAR-10 data file not found", path=str(data_file))

    def _initialize_dataset(self, max_samples: int) -> None:
        """Initialize dataset state and load necessary data."""
        self.max_samples = max_samples
        self._load_meta()
        self._load_data()
        self.length = len(self.images)

    def _load_meta(self) -> None:
        """Load metadata including class names."""
        try:
            with open(self.meta_file, 'rb') as f:
                meta_dict = pickle.load(f, encoding='bytes')
                # Handle both byte and string keys
                if b'label_names' in meta_dict:
                    self.classes = [label.decode('utf-8') if isinstance(label, bytes) else label 
                                  for label in meta_dict[b'label_names']]
                elif 'label_names' in meta_dict:
                    self.classes = meta_dict['label_names']
                else:
                    # Default CIFAR-10 classes if not found in meta
                    self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                                  'dog', 'frog', 'horse', 'ship', 'truck']
                    logger.warning("Using default CIFAR-10 class names")
        except Exception as e:
            logger.error(f"Error loading metadata from {self.meta_file}: {e}")
            # Use default classes as fallback
            self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']
            logger.warning("Failed to load metadata, using default class names")

    def _load_data(self) -> None:
        """Load CIFAR-10 images and labels from pickle files."""
        images = []
        labels = []
        
        for data_file in self.data_files:
            try:
                with open(data_file, 'rb') as f:
                    batch_dict = pickle.load(f, encoding='bytes')
                    
                    # Handle both byte and string keys
                    if b'data' in batch_dict:
                        batch_data = batch_dict[b'data']
                        batch_labels = batch_dict.get(b'labels', batch_dict.get(b'fine_labels', []))
                    else:
                        batch_data = batch_dict['data']
                        batch_labels = batch_dict.get('labels', batch_dict.get('fine_labels', []))
                    
                    # CIFAR-10 data is stored as (N, 3072) where 3072 = 32*32*3
                    # Reshape to (N, 3, 32, 32) and transpose to (N, 32, 32, 3)
                    batch_images = batch_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                    
                    images.append(batch_images)
                    labels.extend(batch_labels)
                    
            except Exception as e:
                logger.error(f"Error loading data from {data_file}: {e}")
                raise DatasetProcessingError(f"Failed to load data file: {str(e)}")
        
        # Concatenate all batches
        self.images = np.vstack(images)
        self.labels = np.array(labels)
        
        # Apply max_samples limit if specified
        if self.max_samples > 0 and self.max_samples < len(self.images):
            self.images = self.images[:self.max_samples]
            self.labels = self.labels[:self.max_samples]
        
        logger.info(f"Loaded {len(self.images)} CIFAR-10 {'training' if self.train else 'test'} samples")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """Get image, label, and filename for given index."""
        self._validate_index(index)
        
        try:
            # Get image and label
            image_array = self.images[index]
            label = int(self.labels[index])
            
            # Convert numpy array to PIL Image
            image = Image.fromarray(image_array.astype(np.uint8), mode='RGB')
            
            # Apply transform
            if self.transform:
                from .core.transforms import ImageTransformer
                image = ImageTransformer.apply_transform(image, self.transform)
            
            # Apply target transform if specified
            if self.target_transform:
                label = self.target_transform(label)
            
            # Generate filename
            class_name = self.classes[label] if label < len(self.classes) else f"class_{label}"
            filename = f"cifar10_{'train' if self.train else 'test'}_{index:05d}_{class_name}.png"
            
            return image, label, filename
            
        except Exception as e:
            logger.error(f"Error processing CIFAR-10 sample {index}: {e}")
            if isinstance(e, DatasetTransformError):
                raise
            raise DatasetProcessingError(f"Failed to process sample {index}: {str(e)}")

    def get_original_image(self, image_file: str) -> Optional[Image.Image]:
        """Load original untransformed image by filename."""
        try:
            # Parse the filename to get the index
            parts = image_file.split('_')
            if len(parts) >= 3 and parts[0] == 'cifar10':
                index = int(parts[2])
                if 0 <= index < len(self.images):
                    image_array = self.images[index]
                    return Image.fromarray(image_array.astype(np.uint8), mode='RGB')
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse filename {image_file}: {e}")
        return None


def load_cifar10_dataset(
    root: Union[str, Path],
    train: bool = True,
    transform: Optional[Callable] = None,
    max_samples: int = -1,
    **kwargs,
) -> CIFAR10Dataset:
    """Factory function to create a CIFAR10Dataset.
    
    Args:
        root: Root directory containing CIFAR-10 data
        train: Whether to load training (True) or test (False) data
        transform: Transform to apply to images (uses CIFAR default if None)
        max_samples: Maximum number of samples to load (-1 for all)
        **kwargs: Additional arguments passed to CIFAR10Dataset
        
    Returns:
        CIFAR10Dataset instance
    """
    if transform is None:
        # CIFAR-10 uses 32x32 RGB images, create appropriate transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    return CIFAR10Dataset(
        root=root,
        train=train,
        transform=transform,
        max_samples=max_samples,
        **kwargs
    ) 