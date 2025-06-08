"""
Tensor compression utilities for network transmission in split computing.

This module provides specialized compression tools for neural network tensors
to optimize network transmission in distributed computation environments.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, cast

import blosc2  # type: ignore
import logging
import pickle
import socket

from .protocols import (
    LENGTH_PREFIX_SIZE,
    CHUNK_SIZE,
    HIGHEST_PROTOCOL,
)
from ..core import NetworkError

# Import tensor encryption only if available
try:
    from .encryption import TensorEncryption
    ENCRYPTION_AVAILABLE = True
except ImportError:
    TensorEncryption = None
    ENCRYPTION_AVAILABLE = False

logger = logging.getLogger("split_computing_logger")


@dataclass(frozen=True)
class CompressionConfig:
    """Configuration settings for tensor compression optimization."""

    clevel: int  # Compression level (0=fast/low, 9=slow/high)
    filter: str  # Data preparation filter (e.g., "NOSHUFFLE", "SHUFFLE", "BITSHUFFLE")
    codec: str  # Compression algorithm (e.g., "ZSTD", "LZ4", "BLOSCLZ")

    def __post_init__(self) -> None:
        """Validate compression configuration parameters for tensor optimization."""
        if not 0 <= self.clevel <= 9:
            raise ValueError("Compression level must be between 0 and 9")

        if self.filter not in blosc2.Filter.__members__:
            raise ValueError(f"Invalid filter: {self.filter}")

        if self.codec not in blosc2.Codec.__members__:
            raise ValueError(f"Invalid codec: {self.codec}")


class CompressionError(Exception):
    """Base exception for tensor compression-related errors."""

    pass


class DecompressionError(CompressionError):
    """Exception raised when tensor decompression fails."""

    pass


class DataCompression:
    """Handles advanced tensor compression for distributed neural network computation."""

    def __init__(self, config: Dict[str, Any], encryption: Optional[TensorEncryption] = None, encryption_mode: str = "transmission") -> None:
        """
        Initialize compression and encryption engine with configuration.
        
        Args:
            config: Compression configuration
            encryption: Optional encryption module for secure tensor transmission
            encryption_mode: Mode for encryption handling ("transmission" or "full")
        """
        self.config = CompressionConfig(
            clevel=config.get("clevel", 3),
            filter=config.get("filter", "NOSHUFFLE"),
            codec=config.get("codec", "ZSTD"),
        )
        # Map string parameters to actual blosc2 enum values for direct API use
        self._filter = blosc2.Filter[self.config.filter]
        self._codec = blosc2.Codec[self.config.codec]
        
        # Store encryption module if provided
        self.encryption = encryption
        self.encryption_enabled = encryption is not None
        self.encryption_mode = encryption_mode

    def compress_data(self, data: Any) -> Tuple[bytes, int]:
        """
        Compress and optionally encrypt tensor data for network transmission.

        === TENSOR SHARING - COMPRESSION PHASE ===
        Pipeline implemented (encrypt->serialize->compress):
        1. If encryption is enabled, encrypt the tensor component first using TenSEAL
           (regardless of mode). For "transmission" mode the server will decrypt
           before computation; for "full" mode the tensor remains encrypted end-to-end.
        2. Package the encrypted tensor together with required metadata so that the
           receiver knows how to post-process it (original tensor size, etc.).
        3. Serialize the package with pickle using the highest protocol.
        4. Compress the serialized bytes with Blosc2 for efficient transfer.
        """
        try:
            # -------------------------------------------------------------
            # STEP 1: build the payload to be serialized
            # -------------------------------------------------------------
            if self.encryption_enabled:
                import torch  # Local import to avoid mandatory torch when unused
                # We only attempt to encrypt if the incoming payload matches the
                # expected (tensor, original_size) tuple produced by the pipeline.
                if isinstance(data, tuple) and len(data) == 2:
                    tensor_data, original_size = data
                    if isinstance(tensor_data, torch.Tensor):
                        # Encrypt tensor using helper utility
                        from ..utils.tensor_encryption import encrypt_tensor
                        encrypted_tensor_data = encrypt_tensor(tensor_data, self.encryption)

                        # Mark payload with encryption mode for downstream logic
                        payload: Dict[str, Any] = {
                            "encrypted_tensor": encrypted_tensor_data,
                            "original_size": original_size,
                            "is_encrypted": True,
                            "encryption_mode": self.encryption.mode,
                        }
                    else:
                        # Not a tensor—fallback to regular serialization path
                        payload = data
                else:
                    # Non-standard payload (e.g. inference result). Do not encrypt – just forward.
                    payload = data
            else:
                payload = data

            # -------------------------------------------------------------
            # STEP 2: serialize & STEP 3: compress
            # -------------------------------------------------------------
            serialized_data = pickle.dumps(payload, protocol=HIGHEST_PROTOCOL)

            compressed_data = blosc2.compress(
                serialized_data,
                clevel=self.config.clevel,
                filter=self._filter,
                codec=self._codec,
            )

            # Return compressed bytes and their true length (after compression)
            return compressed_data, len(compressed_data)
        except Exception as e:
            logger.error(f"Tensor compression failed: {e}")
            raise CompressionError(f"Failed to compress tensor data: {e}")

    def decompress_data(self, data: bytes) -> Any:
        """
        Decompress data and, depending on encryption mode, optionally decrypt it.

        For "transmission" mode we decrypt here on the server side so that
        downstream model receives a plain PyTorch tensor.
        For "full" mode we keep the encrypted package intact; the model (or
        subsequent pipeline stage) must be able to operate on encrypted data.
        """
        try:
            # First decompress using Blosc2
            decompressed_data = blosc2.decompress(data)

            # If encryption disabled just deserialize and return
            if not self.encryption_enabled:
                return pickle.loads(decompressed_data)

            # Attempt to deserialize – may raise if data is not pickle-encoded
            deserialized = pickle.loads(decompressed_data)

            # Check for our encryption package structure
            if isinstance(deserialized, dict) and deserialized.get("is_encrypted"):
                mode = deserialized.get("encryption_mode", "transmission")
                if mode == "transmission":
                    # Decrypt immediately and return plain tensor tuple
                    return self.decrypt_encrypted_package(deserialized)
                elif mode == "full":
                    # Return encrypted package in expected (tensor_data, original_size) format
                    # The encrypted_tensor will be processed by homomorphic-aware components
                    encrypted_tensor_data = deserialized["encrypted_tensor"]
                    original_size = deserialized["original_size"]
                    return (encrypted_tensor_data, original_size)
                else:
                    logger.warning(f"Unknown encryption_mode '{mode}', returning raw data")
                    return deserialized
            else:
                # Not an encrypted package – return as-is
                return deserialized
        except Exception as e:
            logger.error(f"Tensor decompression failed: {e}")
            raise DecompressionError(f"Failed to process received tensor data: {e}")

    def decrypt_encrypted_package(self, encrypted_package: Dict[str, Any]) -> Any:
        """
        Decrypt an encrypted tensor package produced by `compress_data`.

        Returns the standard `(tensor, original_size)` tuple expected by the
        downstream pipeline.
        """
        if not encrypted_package.get("is_encrypted"):
            raise ValueError("Package is not encrypted")

        mode = encrypted_package.get("encryption_mode", "transmission")

        from ..utils.tensor_encryption import decrypt_tensor

        encrypted_tensor_data = encrypted_package["encrypted_tensor"]
        original_size = encrypted_package["original_size"]

        # Transmission mode → decrypt to plain tensor
        if mode == "transmission":
            decrypted_tensor = decrypt_tensor(encrypted_tensor_data, self.encryption)
            return (decrypted_tensor, original_size)
        # Full mode → return the encrypted package (server should not decrypt)
        elif mode == "full":
            return encrypted_package  # Caller decides how to handle
        else:
            raise ValueError(f"Unknown encryption mode: {mode}")

    @staticmethod
    def _receive_chunk(conn: socket.socket, size: int) -> bytes:
        """
        Receive a specific sized chunk of tensor data from a socket.

        This internal method provides reliable data reception by checking for
        socket disconnections during tensor transfer.
        """
        chunk = conn.recv(size)
        if not chunk:
            # Empty response indicates closed connection
            raise NetworkError("Socket connection broken during tensor transmission")
        return chunk

    def receive_full_message(self, conn: socket.socket, expected_length: int) -> bytes:
        """
        Receive complete tensor data of specified length from network connection.

        === TENSOR SHARING - RECEPTION PHASE ===
        Handles large tensor reception by:
        1. Determining if the tensor fits in a single network packet
        2. For larger tensors, receiving and assembling multiple chunks
        3. Ensuring all bytes are received completely before processing

        This method is critical for reliable tensor transmission as deep learning
        tensors can easily exceed single packet sizes.
        """
        if expected_length <= CHUNK_SIZE:
            # Small tensor can be received in one operation
            return self._receive_chunk(conn, expected_length)

        # Allocate space for the complete tensor data
        data_chunks = bytearray(expected_length)
        bytes_received = 0

        # Receive tensor in chunks until complete
        while bytes_received < expected_length:
            remaining = expected_length - bytes_received
            chunk_size = min(remaining, CHUNK_SIZE)

            try:
                # Get next chunk of tensor data
                chunk = self._receive_chunk(conn, chunk_size)

                # Insert chunk at the correct position in the buffer
                data_chunks[bytes_received : bytes_received + len(chunk)] = chunk
                bytes_received += len(chunk)
            except Exception as e:
                raise NetworkError(f"Failed to receive tensor data: {e}")

        # Convert to immutable bytes before returning
        return bytes(data_chunks)

    def receive_data(self, conn: socket.socket) -> Optional[Dict[str, Any]]:
        """
        Receive and decompress tensor data with length-prefixed framing.

        === TENSOR SHARING - COMPLETE RECEPTION PIPELINE ===
        This method handles the complete tensor reception process for distributed
        neural network computation by:
        1. Receiving the length prefix to know tensor size
        2. Receiving the complete tensor data payload
        3. Decompressing the received tensor data back to usable form

        This is the entry point for receiving tensor data from network clients.
        """
        try:
            # Read length prefix to get tensor data size
            length_data = self._receive_chunk(conn, LENGTH_PREFIX_SIZE)
            data_length = int.from_bytes(length_data, byteorder="big")

            # Receive complete tensor data payload
            compressed_data = self.receive_full_message(conn, data_length)

            # Decompress the received tensor data
            decompressed_result = self.decompress_data(compressed_data)

            logger.debug(f"Successfully received and decompressed {data_length} bytes")
            return decompressed_result

        except Exception as e:
            logger.error(f"Failed to receive tensor data: {e}")
            return None

    def send_result(self, conn: socket.socket, result: Any) -> None:
        """
        Compress and transmit tensor result back to network client.

        === TENSOR SHARING - RESULT TRANSMISSION ===
        Handles the return path in tensor sharing by:
        1. Compressing the processed tensor result
        2. Sending the result size as a length prefix
        3. Transmitting the compressed result data

        This completes the tensor sharing cycle by returning processed data.
        """
        try:
            # Compress the result tensor for transmission
            compressed_result, result_size = self.compress_data(result)

            # Send length prefix followed by compressed result data
            length_prefix = result_size.to_bytes(LENGTH_PREFIX_SIZE, byteorder="big")
            conn.sendall(length_prefix + compressed_result)

            logger.debug(f"Successfully sent {result_size} bytes of compressed result")

        except Exception as e:
            logger.error(f"Failed to send tensor result: {e}")
            raise NetworkError(f"Failed to transmit result: {e}")
