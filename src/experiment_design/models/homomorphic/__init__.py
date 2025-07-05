"""Homomorphic Encryption Wrappers for Neural Networks

This package provides a structured approach to implementing homomorphic encryption
capabilities for different neural network architectures. Each network has its own
specialized wrapper that extends from a common base class.

Usage:
    from src.experiment_design.models.homomorphic import get_homomorphic_wrapper
    
    wrapper = get_homomorphic_wrapper("mnist_convnet", model_instance, encryption_context)
    result = wrapper.homomorphic_forward(encrypted_tensor, metadata)
"""

from .base import BaseHomomorphicWrapper, HomomorphicWrapperRegistry
from .mnist_convnet_wrapper import MNISTConvNetHomomorphicWrapper
from .cifar_convnet_wrapper import CIFARConvNetHomomorphicWrapper

# Import wrappers to ensure they are registered
from . import mnist_convnet_wrapper
from . import cifar_convnet_wrapper

def get_homomorphic_wrapper(model_name: str, model_instance, encryption_context):
    """Factory function to get the appropriate homomorphic wrapper for a model.
    
    Args:
        model_name: Name of the model (e.g., "mnist_convnet", "cifar_convnet")
        model_instance: The actual model instance to wrap
        encryption_context: TenSEAL encryption context
        
    Returns:
        BaseHomomorphicWrapper: Appropriate wrapper instance for the model
        
    Raises:
        ValueError: If no wrapper is registered for the model name
    """
    return HomomorphicWrapperRegistry.get_wrapper(model_name, model_instance, encryption_context)

__all__ = [
    "BaseHomomorphicWrapper",
    "HomomorphicWrapperRegistry", 
    "MNISTConvNetHomomorphicWrapper",
    "CIFARConvNetHomomorphicWrapper",
    "get_homomorphic_wrapper"
] 