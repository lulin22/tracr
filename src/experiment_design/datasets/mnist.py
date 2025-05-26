"""MNIST dataset implementation"""

import gzip
import logging
import struct
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from .core import (
    BaseDataset,
    DatasetPathError,
    DatasetProcessingError,
    DatasetTransformError,
    TransformFactory,
    TransformType,
)

logger = logging.getLogger("split_computing_logger")


class MNISTDataset(BaseDataset):
    """Dataset implementation for loading and processing MNIST data."""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_samples: int = -1,
    ) -> None:
        """Initialize MNIST dataset with specified parameters."""
        super().__init__(root, transform, target_transform, max_samples)
        self.train = train
        self._initialize_paths(root)
        if not self.transform:
            self.transform = TransformFactory.get_transform(TransformType.MNIST)
        self._initialize_dataset(max_samples)

    def _initialize_paths(self, root: Optional[Union[str, Path]]) -> None:
        """Set up dataset paths and verify existence."""
        if root is None:
            self.root = self.data_source_dir / "MNIST"
        else:
            self.root = Path(root)
        
        if not self.root.exists():
            logger.error(f"MNIST root directory not found: {self.root}")
            raise DatasetPathError("MNIST root directory not found", path=str(self.root))
        
        self.raw_folder = self.root / "raw"
        if not self.raw_folder.exists():
            logger.error(f"MNIST raw data directory not found: {self.raw_folder}")
            raise DatasetPathError("MNIST raw data directory not found", path=str(self.raw_folder))

    def _initialize_dataset(self, max_samples: int) -> None:
        """Initialize dataset state and load necessary data."""
        self.max_samples = max_samples
        self.classes = [str(i) for i in range(10)]
        self._load_data()
        self.length = len(self.images)

    def _load_data(self) -> None:
        """Load MNIST images and labels from binary files."""
        if self.train:
            images_file = "train-images-idx3-ubyte"
            labels_file = "train-labels-idx1-ubyte"
        else:
            images_file = "t10k-images-idx3-ubyte"
            labels_file = "t10k-labels-idx1-ubyte"
        
        # Load images - try multiple naming conventions and locations
        images_path = self.raw_folder / images_file
        if not images_path.exists():
            images_path = self.raw_folder / f"{images_file}.gz"
            if not images_path.exists():
                # Try alternative naming convention with dots
                alt_images_file = images_file.replace("-idx3-ubyte", ".idx3-ubyte")
                images_path = self.raw_folder / alt_images_file
                if not images_path.exists():
                    images_path = self.raw_folder / f"{alt_images_file}.gz"
                    if not images_path.exists():
                        raise DatasetPathError(f"MNIST images file not found: {images_file}")
        
        # If it's a directory, look for the file inside it
        if images_path.is_dir():
            images_path = images_path / images_file
            if not images_path.exists():
                # Try with dot naming inside directory
                alt_images_file = images_file.replace("-idx3-ubyte", ".idx3-ubyte")
                images_path = images_path.parent / alt_images_file
                if not images_path.exists():
                    raise DatasetPathError(f"MNIST images file not found inside directory: {images_file}")
        
        self.images = self._read_image_file(images_path)
        
        # Load labels - try multiple naming conventions and locations
        labels_path = self.raw_folder / labels_file
        if not labels_path.exists():
            labels_path = self.raw_folder / f"{labels_file}.gz"
            if not labels_path.exists():
                # Try alternative naming convention with dots
                alt_labels_file = labels_file.replace("-idx1-ubyte", ".idx1-ubyte")
                labels_path = self.raw_folder / alt_labels_file
                if not labels_path.exists():
                    labels_path = self.raw_folder / f"{alt_labels_file}.gz"
                    if not labels_path.exists():
                        raise DatasetPathError(f"MNIST labels file not found: {labels_file}")
        
        # If it's a directory, look for the file inside it
        if labels_path.is_dir():
            labels_path = labels_path / labels_file
            if not labels_path.exists():
                # Try with dot naming inside directory
                alt_labels_file = labels_file.replace("-idx1-ubyte", ".idx1-ubyte")
                labels_path = labels_path.parent / alt_labels_file
                if not labels_path.exists():
                    raise DatasetPathError(f"MNIST labels file not found inside directory: {labels_file}")
        
        self.labels = self._read_label_file(labels_path)
        
        # Apply max_samples limit if specified
        if self.max_samples > 0 and self.max_samples < len(self.images):
            self.images = self.images[:self.max_samples]
            self.labels = self.labels[:self.max_samples]

    def _read_image_file(self, path: Path) -> np.ndarray:
        """Read MNIST image file in IDX format."""
        try:
            if path.suffix == '.gz':
                with gzip.open(path, 'rb') as f:
                    data = f.read()
            else:
                with open(path, 'rb') as f:
                    data = f.read()
            
            magic, num_images, rows, cols = struct.unpack('>IIII', data[:16])
            if magic != 2051:
                raise DatasetProcessingError(f"Invalid magic number in image file: {magic}")
            
            images = np.frombuffer(data[16:], dtype=np.uint8)
            return images.reshape(num_images, rows, cols)
        except Exception as e:
            logger.error(f"Error reading image file {path}: {e}")
            raise DatasetProcessingError(f"Failed to read image file: {str(e)}")

    def _read_label_file(self, path: Path) -> np.ndarray:
        """Read MNIST label file in IDX format."""
        try:
            if path.suffix == '.gz':
                with gzip.open(path, 'rb') as f:
                    data = f.read()
            else:
                with open(path, 'rb') as f:
                    data = f.read()
            
            magic, num_labels = struct.unpack('>II', data[:8])
            if magic != 2049:
                raise DatasetProcessingError(f"Invalid magic number in label file: {magic}")
            
            return np.frombuffer(data[8:], dtype=np.uint8)
        except Exception as e:
            logger.error(f"Error reading label file {path}: {e}")
            raise DatasetProcessingError(f"Failed to read label file: {str(e)}")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """Get image, label, and filename for given index."""
        self._validate_index(index)
        
        try:
            image_array = self.images[index]
            label = int(self.labels[index])
            
            # Keep as grayscale for ConvNet model (expects 1 channel)
            image = Image.fromarray(image_array, mode='L')
            
            # Apply transform
            if self.transform:
                from .core.transforms import ImageTransformer
                image = ImageTransformer.apply_transform(image, self.transform)
            
            filename = f"mnist_{'train' if self.train else 'test'}_{index:05d}_{label}.png"
            return image, label, filename
        except Exception as e:
            logger.error(f"Error processing MNIST sample {index}: {e}")
            if isinstance(e, DatasetTransformError):
                raise
            raise DatasetProcessingError(f"Failed to process sample {index}: {str(e)}")

    def get_original_image(self, image_file: str) -> Optional[Image.Image]:
        """Load original untransformed image by filename."""
        try:
            parts = image_file.split('_')
            if len(parts) >= 3 and parts[0] == 'mnist':
                index = int(parts[2])
                if 0 <= index < len(self.images):
                    image_array = self.images[index]
                    # Keep as grayscale for ConvNet model
                    return Image.fromarray(image_array, mode='L')
        except (ValueError, IndexError):
            pass
        return None


def load_mnist_dataset(
    root: Union[str, Path],
    train: bool = True,
    transform: Optional[Callable] = None,
    max_samples: int = -1,
    **kwargs,
) -> MNISTDataset:
    """Factory function to create a MNISTDataset."""
    if transform is None:
        transform = TransformFactory.get_transform(TransformType.MNIST)
    
    return MNISTDataset(
        root=root,
        train=train,
        transform=transform,
        max_samples=max_samples,
        **kwargs
    ) 