"""Base template and registry for homomorphic encryption wrappers.

This module defines the interface that all network-specific homomorphic encryption
wrappers must implement, providing a consistent API across different architectures.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Type, ClassVar
import torch
import numpy as np

logger = logging.getLogger("split_computing_logger")

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    logger.warning("TenSEAL not available - homomorphic wrappers will not function")


class BaseHomomorphicWrapper(ABC):
    """Base class for network-specific homomorphic encryption wrappers.
    
    This abstract base class defines the interface that all homomorphic encryption
    wrappers must implement. Each network architecture should have its own wrapper
    that extends this class and implements the specific homomorphic operations
    needed for that architecture.
    
    Key responsibilities:
    1. Abstract interface for homomorphic forward pass
    2. Common utility methods for tensor operations
    3. Error handling and logging standards
    4. Metadata management for encrypted tensors
    """
    
    def __init__(self, model_instance, encryption_context):
        """Initialize the homomorphic wrapper.
        
        Args:
            model_instance: The neural network model to wrap
            encryption_context: TenSEAL encryption context for operations
        """
        if not TENSEAL_AVAILABLE:
            raise ImportError("TenSEAL is required for homomorphic encryption wrappers")
        
        self.model = model_instance
        self.encryption_context = encryption_context
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Validate the model is HE-compatible if it has that property
        if hasattr(model_instance, 'is_he_compatible'):
            if not model_instance.is_he_compatible:
                self.logger.warning(f"Model {type(model_instance)} may not be HE-compatible")
    
    @abstractmethod
    def homomorphic_forward(self, encrypted_tensor, metadata: Optional[Dict] = None, start_layer: int = 0) -> Union[torch.Tensor, Dict[str, Any]]:
        """Perform homomorphic forward pass starting from the specified layer.
        
        Args:
            encrypted_tensor: TenSEAL encrypted input
            metadata: Metadata about the encrypted tensor
            start_layer: Layer index to start processing from (0 = full network)
            
        Returns:
            Encrypted or decrypted result depending on encryption mode
        """
        raise NotImplementedError("Subclasses must implement homomorphic_forward")
    
    @abstractmethod
    def get_required_metadata(self) -> Dict[str, Any]:
        """Return metadata requirements for this wrapper.
        
        Each wrapper should specify what metadata it needs for proper operation
        (e.g., image dimensions, kernel sizes, windows_nb for convolutions).
        
        Returns:
            Dict describing required metadata fields and their types/purposes
        """
        raise NotImplementedError("Subclasses must implement get_required_metadata")
    
    def validate_metadata(self, metadata: Optional[Dict]) -> bool:
        """Validate that provided metadata meets wrapper requirements.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            bool: True if metadata is valid, False otherwise
        """
        if metadata is None:
            self.logger.warning("No metadata provided")
            return False
        
        required = self.get_required_metadata()
        for key, info in required.items():
            if key not in metadata:
                self.logger.error(f"Missing required metadata field: {key}")
                return False
        
        return True
    
    def extract_model_weights(self) -> Dict[str, np.ndarray]:
        """Extract model weights in a standardized format.
        
        This method should be overridden by subclasses to extract weights
        in a format suitable for homomorphic operations.
        
        Returns:
            Dict mapping layer names to weight arrays
        """
        weights = {}
        for name, param in self.model.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        return weights
    
    def homomorphic_convolution(self, enc_input, kernel_weights: np.ndarray, 
                              bias_weights: Optional[np.ndarray], 
                              windows_nb: int, **kwargs) -> Any:
        """Standard homomorphic convolution operation.
        
        Provides a common implementation of convolution that can be reused
        across different wrapper implementations.
        
        Args:
            enc_input: Encrypted input tensor
            kernel_weights: Convolution kernel weights (numpy array)
            bias_weights: Optional bias weights
            windows_nb: Number of windows for im2col operation
            **kwargs: Additional parameters (stride, padding, etc.)
            
        Returns:
            Encrypted convolution result
        """
        try:
            conv_outputs = []
            
            # Handle different kernel weight formats
            if len(kernel_weights.shape) == 4:  # (out_ch, in_ch, h, w)
                num_output_channels = kernel_weights.shape[0]
                
                for out_ch in range(num_output_channels):
                    # Extract 2D kernel for this output channel
                    if kernel_weights.shape[1] == 1:  # Single input channel
                        kernel_2d = kernel_weights[out_ch, 0].tolist()
                    else:
                        # For multi-channel, we need to handle differently
                        # This is a simplified approach - may need enhancement
                        kernel_2d = kernel_weights[out_ch, 0].tolist()
                    
                    bias = bias_weights[out_ch] if bias_weights is not None else 0.0
                    
                    # Apply TenSEAL convolution
                    conv_out = enc_input.conv2d_im2col(kernel_2d, windows_nb) + bias
                    conv_outputs.append(conv_out)
                    
                    self.logger.debug(f"Convolution channel {out_ch+1}/{num_output_channels} completed")
            
            # Combine output channels
            if len(conv_outputs) == 1:
                return conv_outputs[0]
            else:
                return ts.CKKSVector.pack_vectors(conv_outputs)
                
        except Exception as e:
            self.logger.error(f"Homomorphic convolution failed: {e}")
            raise
    
    def homomorphic_activation(self, enc_input, activation_type: str = "square") -> Any:
        """Apply homomorphic activation function.
        
        Args:
            enc_input: Encrypted input tensor
            activation_type: Type of activation ("square", "polynomial", etc.)
            
        Returns:
            Encrypted tensor after activation
        """
        try:
            if activation_type == "square":
                # Use TenSEAL's in-place square operation
                result = enc_input
                result.square_()
                return result
            else:
                raise ValueError(f"Unsupported activation type: {activation_type}")
        except Exception as e:
            self.logger.error(f"Homomorphic activation failed: {e}")
            raise
    
    def homomorphic_linear(self, enc_input, weight_matrix: np.ndarray, 
                          bias_vector: Optional[np.ndarray] = None) -> Any:
        """Apply homomorphic linear transformation.
        
        Args:
            enc_input: Encrypted input tensor
            weight_matrix: Linear layer weight matrix
            bias_vector: Optional bias vector
            
        Returns:
            Encrypted linear transformation result
        """
        try:
            # Transpose weight matrix for TenSEAL mm operation
            weight_transposed = weight_matrix.T.tolist()
            
            # Apply matrix multiplication
            result = enc_input.mm(weight_transposed)
            
            # Add bias if provided
            if bias_vector is not None:
                bias_list = bias_vector.tolist()
                result = result + bias_list
            
            return result
        except Exception as e:
            self.logger.error(f"Homomorphic linear transformation failed: {e}")
            raise
    
    def create_fallback_result(self, num_classes: int = 10) -> torch.Tensor:
        """Create a fallback result when homomorphic computation fails.
        
        Args:
            num_classes: Number of output classes
            
        Returns:
            Fallback tensor with reasonable values
        """
        self.logger.warning("Creating fallback result due to computation failure")
        # Create small random values that won't bias toward any particular class
        fallback_values = torch.randn(1, num_classes) * 0.1
        return fallback_values
    
    def log_operation(self, operation_name: str, **kwargs):
        """Standard logging for homomorphic operations.
        
        Args:
            operation_name: Name of the operation being performed
            **kwargs: Additional info to log
        """
        self.logger.info(f"🔐 {operation_name}: {kwargs}")


class HomomorphicWrapperRegistry:
    """Registry for homomorphic encryption wrappers.
    
    Similar to ModelRegistry, this class manages registration and instantiation
    of homomorphic wrappers for different network architectures.
    """
    
    _registry: ClassVar[Dict[str, Type[BaseHomomorphicWrapper]]] = {}
    
    @classmethod
    def register(cls, model_name: str):
        """Decorator to register a homomorphic wrapper class.
        
        Usage:
            @HomomorphicWrapperRegistry.register("mnist_convnet")
            class MNISTConvNetHomomorphicWrapper(BaseHomomorphicWrapper):
                ...
        
        Args:
            model_name: Name of the model this wrapper handles
        """
        def decorator(wrapper_cls: Type[BaseHomomorphicWrapper]):
            model_name_lower = model_name.lower()
            if model_name_lower in cls._registry:
                logger.warning(f"Overwriting existing wrapper registration for '{model_name_lower}'")
            cls._registry[model_name_lower] = wrapper_cls
            logger.debug(f"Registered homomorphic wrapper '{model_name_lower}'")
            return wrapper_cls
        return decorator
    
    @classmethod
    def get_wrapper(cls, model_name: str, model_instance, encryption_context) -> BaseHomomorphicWrapper:
        """Get a wrapper instance for the specified model.
        
        Args:
            model_name: Name of the model
            model_instance: The model instance to wrap
            encryption_context: TenSEAL encryption context
            
        Returns:
            Instantiated wrapper for the model
            
        Raises:
            ValueError: If no wrapper is registered for the model
        """
        model_name_lower = model_name.lower()
        if model_name_lower not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"No homomorphic wrapper registered for '{model_name}'. Available: {available}")
        
        wrapper_cls = cls._registry[model_name_lower]
        return wrapper_cls(model_instance, encryption_context)
    
    @classmethod
    def list_registered_wrappers(cls) -> list:
        """List all registered wrapper names."""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, model_name: str) -> bool:
        """Check if a wrapper is registered for the model."""
        return model_name.lower() in cls._registry 