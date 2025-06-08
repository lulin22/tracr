"""Datasets package for experiment design"""

# Dataset implementations
from .imagenet import ImageNetDataset, load_imagenet_dataset
from .onion import OnionDataset, load_onion_dataset
from .mnist import MNISTDataset, load_mnist_dataset
from .cifar10 import CIFAR10Dataset, load_cifar10_dataset
from .custom import (
    CustomDataset,
    load_custom_dataset,
    custom_collate,
    create_custom_transform,
)

__all__ = [
    # Standard dataset implementations
    "ImageNetDataset",
    "OnionDataset",
    "load_imagenet_dataset",
    "load_onion_dataset",
    "MNISTDataset",
    "load_mnist_dataset",
    "CIFAR10Dataset",
    "load_cifar10_dataset",
    # Custom dataset implementation
    "CustomDataset",
    "load_custom_dataset",
    "custom_collate",
    "create_custom_transform",
]
