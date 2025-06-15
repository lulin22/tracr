"""
Utility functions for tensor encryption and decryption.

This module provides high-level functions for encrypting and decrypting
PyTorch tensors in the split computing framework.
"""

import logging
import torch
from typing import Dict, Any, Union
import numpy as np

logger = logging.getLogger("split_computing_logger")

# Import the actual encryption implementation
try:
    from ..network.encryption import TensorEncryption
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    logger.warning("TensorEncryption not available - using placeholder implementations")


def encrypt_tensor(tensor: torch.Tensor, encryption_instance: 'TensorEncryption' = None) -> Dict[str, Any]:
    """
    Encrypt a PyTorch tensor for secure transmission.
    
    Args:
        tensor: PyTorch tensor to encrypt
        encryption_instance: Optional TensorEncryption instance to use
        
    Returns:
        Dictionary containing encrypted data and metadata
    """
    try:
        if ENCRYPTION_AVAILABLE and encryption_instance:
            # Use actual homomorphic encryption
            encrypted_data, metadata = encryption_instance.encrypt(tensor)
            
            # Ensure shape is included in metadata
            if "shape" not in metadata:
                metadata["shape"] = list(tensor.shape)
            
            # Return in expected format
            return {
                "encrypted_data": encrypted_data,
                "metadata": metadata,
                "is_encrypted": True,
                "encryption_type": "homomorphic",
                "shape": list(tensor.shape),  # Explicitly include shape
                "dtype": str(tensor.dtype)    # Explicitly include dtype
            }
        else:
            # Fallback to placeholder (for testing/development)
            logger.warning("Using placeholder encryption - not cryptographically secure")
            
            # Convert tensor to bytes for encryption
            tensor_numpy = tensor.detach().cpu().numpy()
            tensor_bytes = tensor_numpy.tobytes()
            
            # Store metadata needed for reconstruction
            return {
                "encrypted_data": tensor_bytes,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "is_encrypted": True,
                "encryption_type": "placeholder"
            }
        
    except Exception as e:
        logger.error(f"Failed to encrypt tensor: {e}")
        raise


def decrypt_tensor(encrypted_data: Dict[str, Any], encryption_instance: 'TensorEncryption' = None) -> torch.Tensor:
    """
    Decrypt an encrypted tensor back to PyTorch tensor.
    
    Args:
        encrypted_data: Dictionary containing encrypted tensor data and metadata
        encryption_instance: Optional TensorEncryption instance to use
        
    Returns:
        Reconstructed PyTorch tensor
    """
    try:
        if not isinstance(encrypted_data, dict) or not encrypted_data.get("is_encrypted"):
            raise ValueError("Invalid encrypted data format")
        
        encryption_type = encrypted_data.get("encryption_type", "placeholder")
        
        if ENCRYPTION_AVAILABLE and encryption_instance and encryption_type == "homomorphic":
            # Use actual homomorphic decryption
            decrypted_bytes = encryption_instance.decrypt(
                encrypted_data["encrypted_data"], 
                encrypted_data["metadata"]
            )
            
            # The decrypt method returns pickled tensor bytes
            import pickle
            tensor = pickle.loads(decrypted_bytes)
            
            # Ensure tensor has correct shape and dtype
            if isinstance(tensor, torch.Tensor):
                if "shape" in encrypted_data:
                    tensor = tensor.reshape(encrypted_data["shape"])
                if "dtype" in encrypted_data:
                    tensor = tensor.to(getattr(torch, encrypted_data["dtype"].split(".")[-1]))
            
            logger.debug(f"Decrypted tensor of shape {tensor.shape} and dtype {tensor.dtype}")
            return tensor
        else:
            # Fallback to placeholder decryption
            logger.warning("Using placeholder decryption - not cryptographically secure")
            
            # Extract metadata
            shape = encrypted_data.get("shape")
            dtype_str = encrypted_data.get("dtype", "torch.float32")
            data_bytes = encrypted_data["encrypted_data"]
            
            # Convert string dtype back to torch dtype
            dtype_map = {
                "torch.float32": torch.float32,
                "torch.float64": torch.float64,
                "torch.int32": torch.int32,
                "torch.int64": torch.int64,
                "torch.uint8": torch.uint8,
            }
            torch_dtype = dtype_map.get(dtype_str, torch.float32)
            
            # Reconstruct tensor
            numpy_array = np.frombuffer(data_bytes, dtype=torch_dtype.numpy_dtype if hasattr(torch_dtype, 'numpy_dtype') else np.float32)
            if shape:
                numpy_array = numpy_array.reshape(shape)
            tensor = torch.from_numpy(numpy_array.copy()).to(torch_dtype)
            
            logger.debug(f"Decrypted tensor of shape {tensor.shape} and dtype {tensor.dtype}")
            return tensor
        
    except Exception as e:
        logger.error(f"Failed to decrypt tensor: {e}")
        raise


def encrypt_tensor_homomorphic(tensor: torch.Tensor, encryption_context) -> Dict[str, Any]:
    """
    Encrypt a tensor using homomorphic encryption.
    
    Args:
        tensor: PyTorch tensor to encrypt
        encryption_context: Homomorphic encryption context (TensorEncryption instance)
        
    Returns:
        Dictionary containing encrypted data and metadata
    """
    if isinstance(encryption_context, TensorEncryption):
        return encrypt_tensor(tensor, encryption_context)
    else:
        logger.warning("Invalid encryption context - using placeholder")
        return encrypt_tensor(tensor)


def decrypt_tensor_homomorphic(encrypted_data: Dict[str, Any], encryption_context) -> torch.Tensor:
    """
    Decrypt a homomorphically encrypted tensor.
    
    Args:
        encrypted_data: Dictionary containing encrypted tensor data
        encryption_context: Homomorphic encryption context (TensorEncryption instance)
        
    Returns:
        Reconstructed PyTorch tensor
    """
    if isinstance(encryption_context, TensorEncryption):
        return decrypt_tensor(encrypted_data, encryption_context)
    else:
        logger.warning("Invalid encryption context - using placeholder")
        return decrypt_tensor(encrypted_data) 