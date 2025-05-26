"""
Tensor encryption utilities for secure split computing.

This module provides encryption/decryption capabilities for tensor data
to ensure secure transmission in untrusted networks. It implements homomorphic
encryption using TenSEAL's CKKS scheme to allow computation on encrypted data.
"""

import os
import logging
import pickle
from typing import Tuple, Optional, Dict, Any, Union, List
import torch
import numpy as np

# Import cryptography libraries for key handling
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Import TenSEAL for homomorphic encryption
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    
logger = logging.getLogger("split_computing_logger")


class EncryptionError(Exception):
    """Base exception for encryption-related errors."""
    pass


class DecryptionError(EncryptionError):
    """Exception raised when tensor decryption fails."""
    pass


class KeyManagementError(EncryptionError):
    """Exception raised when key management operations fail."""
    pass


class HomomorphicEncryption:
    """
    Handles homomorphic encryption and decryption of tensor data for secure transmission.

    This class provides a framework for securing tensor data during transmission
    between client and server in split computing architectures. It implements
    homomorphic encryption (CKKS) which allows for computation on encrypted data.
    """

    def __init__(
        self, 
        context: Optional[ts.Context] = None,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: Optional[List[int]] = None,
        global_scale: Optional[int] = None,
        salt: Optional[bytes] = None
    ):
        """
        Initialize the homomorphic encryption module with a TenSEAL context.

        Args:
            context: Optional pre-configured TenSEAL context. If not provided, 
                    a new context will be created with the provided parameters.
            poly_modulus_degree: Polynomial modulus degree (power of 2) for CKKS scheme.
            coeff_mod_bit_sizes: List of bit sizes for coefficient modulus.
            global_scale: Scale to be used for encoding.
            salt: Optional salt for key derivation if using a password.
        """
        if not TENSEAL_AVAILABLE:
            raise ImportError("TenSEAL is not available. Please install it with 'pip install tenseal'")
        
        # Store or generate salt for password-based key derivation
        self.salt = salt if salt is not None else os.urandom(16)
        
        self.encryption_ready = True

        if context is not None:
            # Use provided context
            self.context = context
            logger.info("Using provided TenSEAL context")
        else:
            # Create default context if not provided
            if coeff_mod_bit_sizes is None:
                # Default coefficient modulus bit sizes for CKKS
                bits_scale = 26
                coeff_mod_bit_sizes = [31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
            
            # Create a new context with the CKKS scheme
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes
            )
            
            # Set the global scale if provided
            if global_scale is not None:
                self.context.global_scale = global_scale
            else:
                self.context.global_scale = 2**26  # Default scale
            
            # Generate Galois keys for vector rotation operations
            self.context.generate_galois_keys()
            
            logger.info(f"Created new TenSEAL context with poly_modulus_degree={poly_modulus_degree}")
    
    @classmethod
    def from_password(
        cls, 
        password: str, 
        salt: Optional[bytes] = None,
        poly_modulus_degree: int = 8192
    ) -> "HomomorphicEncryption":
        """
        Create an encryption instance from a password string.

        This method derives parameters from a password using PBKDF2.

        Args:
            password: Password string to derive key from
            salt: Optional salt bytes for key derivation
            poly_modulus_degree: Polynomial modulus degree to use

        Returns:
            Configured HomomorphicEncryption instance

        Raises:
            KeyManagementError: If key derivation fails
        """
        if salt is None:
            salt = os.urandom(16)

        try:
            # Derive parameters using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key_bytes = kdf.derive(password.encode())
            
            # Use the derived key to seed the coefficient modulus bit sizes
            seed = int.from_bytes(key_bytes[:4], byteorder='big')
            np.random.seed(seed)
            
            # Scale factor based on part of the key
            scale_factor = int.from_bytes(key_bytes[4:8], byteorder='big') % 10 + 20
            
            # Generate coefficient modulus bit sizes
            bits_scale = scale_factor
            coeff_mod_bit_sizes = [31]
            # Add a variable number of bit sizes based on the key
            num_scales = (int.from_bytes(key_bytes[8:12], byteorder='big') % 5) + 3
            for _ in range(num_scales):
                coeff_mod_bit_sizes.append(bits_scale)
            coeff_mod_bit_sizes.append(31)
            
            # Create context
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes
            )
            context.global_scale = 2**bits_scale
            context.generate_galois_keys()
            
            return cls(context=context, salt=salt)
        except Exception as e:
            logger.error(f"Key derivation failed: {e}")
            raise KeyManagementError(f"Failed to derive encryption parameters from password: {e}")

    def encrypt(self, data: Union[bytes, np.ndarray, torch.Tensor]) -> Tuple[bytes, dict]:
        """
        Encrypt tensor data using CKKS homomorphic encryption.

        This method:
        1. Converts the input to a format suitable for homomorphic encryption
        2. Encrypts the data using the TenSEAL context
        3. Serializes the encrypted data for transmission

        Args:
            data: Tensor data to encrypt (bytes, numpy array, or PyTorch tensor)

        Returns:
            Tuple of (encrypted_data_bytes, encryption_metadata)

        Raises:
            EncryptionError: If encryption fails
        """
        try:
            # Convert input data to the appropriate format
            if isinstance(data, bytes):
                # If data is already bytes, deserialize it to get the tensor
                tensor_data = pickle.loads(data)
                if isinstance(tensor_data, torch.Tensor):
                    # Convert PyTorch tensor to numpy array
                    numpy_data = tensor_data.detach().cpu().numpy()
                elif isinstance(tensor_data, np.ndarray):
                    numpy_data = tensor_data
                else:
                    numpy_data = np.frombuffer(data, dtype=np.float32)
            elif isinstance(data, torch.Tensor):
                # Convert PyTorch tensor to numpy array
                numpy_data = data.detach().cpu().numpy()
            elif isinstance(data, np.ndarray):
                numpy_data = data
            else:
                raise EncryptionError(f"Unsupported data type for encryption: {type(data)}")
            
            # Flatten the array for encryption
            flat_data = numpy_data.flatten().tolist()
            
            # Encrypt the flattened data
            enc_vector = ts.ckks_vector(self.context, flat_data)
            
            # Serialize the encrypted vector for transmission
            encrypted_data = enc_vector.serialize()
            
            # Save metadata for proper decryption and reshaping
            metadata = {
                "shape": numpy_data.shape,
                "dtype": str(numpy_data.dtype)
            }
            
            return encrypted_data, metadata
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt tensor data: {e}")

    def decrypt(self, encrypted_data: bytes, metadata: dict) -> bytes:
        """
        Decrypt homomorphically encrypted tensor data.

        This method:
        1. Deserializes the encrypted data
        2. Decrypts the data using the TenSEAL context
        3. Reshapes the decrypted data according to the metadata
        4. Returns the decrypted tensor as serialized bytes

        Args:
            encrypted_data: Serialized encrypted tensor data
            metadata: Dictionary containing shape and dtype information

        Returns:
            Decrypted tensor data as pickled bytes

        Raises:
            DecryptionError: If decryption fails
        """
        try:
            # Deserialize the encrypted vector
            enc_vector = ts.ckks_vector_from(self.context, encrypted_data)
            
            # Decrypt the vector
            decrypted_data = enc_vector.decrypt()
            
            # Reshape the data according to metadata
            shape = metadata.get("shape", None)
            dtype_str = metadata.get("dtype", "float32")
            
            # Convert dtype string back to numpy dtype
            dtype = np.dtype(dtype_str)
            
            if shape:
                # Reshape the decrypted data
                numpy_array = np.array(decrypted_data, dtype=dtype).reshape(shape)
            else:
                numpy_array = np.array(decrypted_data, dtype=dtype)
            
            # Convert back to PyTorch tensor if needed
            tensor = torch.from_numpy(numpy_array)
            
            # Serialize the tensor for consistency with the compression pipeline
            return pickle.dumps(tensor)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise DecryptionError(f"Failed to decrypt tensor data: {e}")

    def get_context(self) -> ts.Context:
        """Return the TenSEAL context for serialization or transmission."""
        return self.context

    def serialize_context(self) -> bytes:
        """Serialize the context for transmission."""
        return self.context.serialize()
    
    @classmethod
    def from_serialized_context(cls, serialized_context: bytes) -> "HomomorphicEncryption":
        """Create a HomomorphicEncryption instance from a serialized context."""
        context = ts.context_from(serialized_context)
        return cls(context=context)

    def get_salt(self) -> bytes:
        """Return the salt used for key derivation."""
        return self.salt


class KeyManager:
    """
    Manages encryption keys for secure tensor transmission.

    This class handles context generation, storage, and exchange
    between client and server components.
    """

    def __init__(self, key_directory: Optional[str] = None):
        """
        Initialize key manager with optional key storage directory.

        Args:
            key_directory: Directory to store key files (optional)
        """
        self.key_directory = key_directory
        self.active_contexts: Dict[str, ts.Context] = {}

    def generate_context(self, 
                         key_id: str, 
                         poly_modulus_degree: int = 8192,
                         bit_scale: int = 26) -> ts.Context:
        """
        Generate a new TenSEAL context with the given ID.

        Args:
            key_id: Identifier for the generated context
            poly_modulus_degree: Polynomial modulus degree (power of 2)
            bit_scale: Bit scale for encoding precision

        Returns:
            The generated TenSEAL context
        """
        # Configure coefficient modulus bit sizes
        coeff_mod_bit_sizes = [31, bit_scale, bit_scale, bit_scale, bit_scale, bit_scale, bit_scale, 31]
        
        # Create a new context
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        context.global_scale = 2**bit_scale
        context.generate_galois_keys()

        # Store the context in memory
        self.active_contexts[key_id] = context

        return context

    def load_context(self, key_path: str) -> ts.Context:
        """
        Load TenSEAL context from secure storage.

        Args:
            key_path: Path to the serialized context file

        Returns:
            The loaded TenSEAL context

        Raises:
            KeyManagementError: If context loading fails
        """
        try:
            # Load the serialized context from file
            with open(key_path, 'rb') as key_file:
                serialized_context = key_file.read()
                
            # Deserialize the context
            context = ts.context_from(serialized_context)
            return context
        except Exception as e:
            raise KeyManagementError(f"Failed to load context from {key_path}: {e}")

    def save_context(self, key_id: str, key_path: str) -> None:
        """
        Save a TenSEAL context to secure storage.

        Args:
            key_id: Identifier for the context to save
            key_path: Path where the context should be saved

        Raises:
            KeyManagementError: If context saving fails
        """
        try:
            if key_id not in self.active_contexts:
                raise KeyManagementError(f"Context ID {key_id} not found")
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(key_path), exist_ok=True)
            
            # Serialize and save the context with restricted permissions
            with open(key_path, 'wb') as key_file:
                key_file.write(self.active_contexts[key_id].serialize())
                
            # Set appropriate permissions (POSIX systems only)
            try:
                os.chmod(key_path, 0o600)  # Only owner can read/write
            except:
                logger.warning(f"Could not set file permissions on {key_path}")
                
        except Exception as e:
            raise KeyManagementError(f"Failed to save context to {key_path}: {e}")


def create_encryption(
    password: Optional[str] = None,
    key_file: Optional[str] = None,
    generate_new: bool = False,
    poly_modulus_degree: int = 8192,
    bit_scale: int = 26
) -> HomomorphicEncryption:
    """
    Factory function to create a configured HomomorphicEncryption instance.

    Args:
        password: Optional password to derive encryption parameters from
        key_file: Optional path to load serialized context from
        generate_new: Whether to generate a new random context
        poly_modulus_degree: Polynomial modulus degree for new contexts
        bit_scale: Bit scale for encoding precision

    Returns:
        Configured HomomorphicEncryption instance

    Raises:
        KeyManagementError: If encryption creation fails
    """
    if not TENSEAL_AVAILABLE:
        raise ImportError("TenSEAL is not available. Please install it with 'pip install tenseal'")
        
    try:
        if password:
            logger.info("Creating homomorphic encryption from password")
            return HomomorphicEncryption.from_password(
                password, 
                poly_modulus_degree=poly_modulus_degree
            )

        if key_file:
            logger.info(f"Loading encryption context from {key_file}")
            key_manager = KeyManager()
            context = key_manager.load_context(key_file)
            return HomomorphicEncryption(context=context)

        if generate_new:
            logger.info("Generating new random encryption context")
            # Configure coefficient modulus bit sizes
            coeff_mod_bit_sizes = [31, bit_scale, bit_scale, bit_scale, bit_scale, bit_scale, bit_scale, 31]
            
            # Create context with specified parameters
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes
            )
            context.global_scale = 2**bit_scale
            context.generate_galois_keys()
            
            return HomomorphicEncryption(context=context)

        # Default case
        logger.info("Creating encryption with default settings")
        return HomomorphicEncryption()

    except Exception as e:
        logger.error(f"Failed to create encryption: {e}")
        raise KeyManagementError(f"Failed to create encryption: {e}")

# Alias for backward compatibility
TensorEncryption = HomomorphicEncryption
