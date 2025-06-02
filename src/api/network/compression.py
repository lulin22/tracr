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
from .encryption import HomomorphicEncryption

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

    def __init__(self, config: Dict[str, Any], encryption: Optional[HomomorphicEncryption] = None, encryption_mode: str = "transmission") -> None:
        """
        Initialize compression and encryption engine with configuration.
        
        Args:
            config: Compression configuration
            encryption: Optional encryption module for secure tensor transmission
            encryption_mode: Encryption mode - "transmission" (decrypt on server) or "full" (process encrypted)
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
        self.encryption_mode = encryption_mode.lower() if encryption_mode else "transmission"
        
        # Validate encryption mode
        if self.encryption_mode not in ["transmission", "full"]:
            logger.warning(f"Invalid encryption mode '{encryption_mode}', using 'transmission'")
            self.encryption_mode = "transmission"
        
        if self.encryption_enabled:
            if self.encryption_mode == "full":
                logger.info("Homomorphic encryption enabled in full mode (compute-on-encrypted)")
            else:
                logger.info("Homomorphic encryption enabled in transmission mode")

    def compress_data(self, data: Any) -> Tuple[bytes, int]:
        """
        Compress and optionally encrypt tensor data for network transmission.

        === TENSOR SHARING - COMPRESSION PHASE ===
        Optimizes neural network tensors for network transmission by:
        1. Serializing the tensor data structure with pickle
        2. Applying homomorphic encryption if enabled
        3. Applying compression with tuned parameters

        Returns a tuple of (compressed_bytes, compressed_length)
        """
        try:
            # Serialize tensor to bytes using highest available pickle protocol
            serialized_data = pickle.dumps(data, protocol=HIGHEST_PROTOCOL)

            # Apply encryption if enabled (BEFORE compression)
            if self.encryption_enabled:
                try:
                    # Use homomorphic encryption to encrypt the serialized data
                    encrypted_data, metadata = self.encryption.encrypt(serialized_data)
                    
                    # Serialize metadata for transmission
                    metadata_bytes = pickle.dumps(metadata, protocol=HIGHEST_PROTOCOL)
                    
                    # Combine metadata and encrypted data
                    # Format: [metadata_length (4 bytes)][metadata][encrypted_data]
                    metadata_len = len(metadata_bytes)
                    data_to_compress = metadata_len.to_bytes(4, byteorder='big') + metadata_bytes + encrypted_data
                except Exception as e:
                    logger.error(f"Tensor encryption failed: {e}")
                    raise CompressionError(f"Failed to encrypt tensor data: {e}")
            else:
                # Use original serialized data for compression
                data_to_compress = serialized_data

            # Apply Blosc2 compression with configured parameters optimized for tensors
            compressed_data = blosc2.compress(
                data_to_compress,
                clevel=self.config.clevel,
                filter=self._filter,
                codec=self._codec,
            )
            
            return compressed_data, len(compressed_data)
        except Exception as e:
            logger.error(f"Tensor compression failed: {e}")
            raise CompressionError(f"Failed to compress tensor data: {e}")

    def decompress_data(self, data: bytes, skip_decryption: bool = False) -> Any:
        """
        Decompress and decrypt (if needed) tensor data.

        === TENSOR SHARING - DECOMPRESSION PHASE ===
        Recovers the original tensor structure from compressed network data by:
        1. Applying Blosc2 decompression to restore serialized bytes
        2. Decrypting the data if encryption is enabled (unless skip_decryption is True)
        3. Deserializing the data back to its original tensor structure

        Args:
            data: Compressed data bytes
            skip_decryption: If True and homomorphic_mode is enabled, skip decryption
                           to allow processing on encrypted tensors
        """
        try:
            # First decompress the data
            decompressed_data = blosc2.decompress(data)
            
            # Handle encrypted data if encryption is enabled
            if self.encryption_enabled and not (self.encryption_mode == "full" and skip_decryption):
                # Normal mode: decrypt after decompression
                # Extract metadata length (first 4 bytes)
                metadata_len = int.from_bytes(decompressed_data[:4], byteorder='big')
                
                # Extract metadata
                metadata_bytes = decompressed_data[4:4+metadata_len]
                metadata = pickle.loads(metadata_bytes)
                
                # Extract encrypted data
                encrypted_data = decompressed_data[4+metadata_len:]
                
                # Decrypt the data
                decrypted_data = self.encryption.decrypt(encrypted_data, metadata)
                
                # The result might be bytes or already deserialized tensor
                if isinstance(decrypted_data, bytes):
                    # If still bytes, deserialize it using pickle
                    return pickle.loads(decrypted_data)
                else:
                    # Otherwise return the already deserialized tensor
                    return decrypted_data
            elif self.encryption_enabled and self.encryption_mode == "full" and skip_decryption:
                # Homomorphic mode: return encrypted tensor data with metadata
                # This allows the server to process encrypted tensors
                metadata_len = int.from_bytes(decompressed_data[:4], byteorder='big')
                metadata_bytes = decompressed_data[4:4+metadata_len]
                metadata = pickle.loads(metadata_bytes)
                encrypted_data = decompressed_data[4+metadata_len:]
                
                # Return both encrypted data and metadata for later decryption
                return {
                    'encrypted_data': encrypted_data,
                    'metadata': metadata,
                    'is_encrypted': True
                }
            else:
                # Standard deserialization without encryption
                return pickle.loads(decompressed_data)
                
        except Exception as e:
            logger.error(f"Tensor decompression/decryption failed: {e}")
            raise DecompressionError(f"Failed to process received tensor data: {e}")

    def decrypt_result(self, encrypted_result: Dict[str, Any]) -> Any:
        """
        Decrypt a result that was processed in encrypted form.
        
        This is used in homomorphic mode to decrypt the final result after
        the server has processed the encrypted tensor.
        
        Args:
            encrypted_result: Dictionary containing encrypted_data and metadata
            
        Returns:
            Decrypted tensor result
        """
        if not self.encryption or self.encryption_mode != "full":
            raise CompressionError("decrypt_result called but not in full mode")
            
        try:
            encrypted_data = encrypted_result['encrypted_data']
            metadata = encrypted_result['metadata']
            
            # Decrypt the result
            decrypted_data = self.encryption.decrypt(encrypted_data, metadata)
            
            # Deserialize if needed
            if isinstance(decrypted_data, bytes):
                return pickle.loads(decrypted_data)
            else:
                return decrypted_data
        except Exception as e:
            logger.error(f"Result decryption failed: {e}")
            raise DecryptionError(f"Failed to decrypt result: {e}")

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

        === TENSOR SHARING - COMPLETE RECEPTION SEQUENCE ===
        Implements a reliable tensor reception protocol:
        1. Reads the length prefix to determine tensor size
        2. Receives the complete tensor using chunked transfers if needed
        3. Decompresses and deserializes the tensor data

        Returns the reconstructed tensor data structure or None if reception fails.
        """
        try:
            # First read the length prefix to determine tensor size
            length_data = self._receive_chunk(conn, LENGTH_PREFIX_SIZE)
            expected_length = int.from_bytes(length_data, "big")

            # Receive the complete compressed tensor
            compressed_data = self.receive_full_message(conn, expected_length)

            # Decompress and return the tensor data
            return cast(Dict[str, Any], self.decompress_data(compressed_data))
        except Exception as e:
            logger.error(f"Error receiving tensor data: {e}")
            return None

    def send_result(self, conn: socket.socket, result: Any) -> None:
        """
        Compress and send tensor result data over network connection.

        === TENSOR SHARING - TRANSMISSION PHASE ===
        Implements reliable tensor transmission protocol:
        1. Compresses the tensor result
        2. Sends the tensor size as a length prefix (for proper framing)
        3. Sends the compressed tensor data

        This method is used for transmitting processed tensor results back to clients.
        """
        try:
            # Compress the tensor result
            compressed, size = self.compress_data(result)

            # Send length prefix first for proper framing
            conn.sendall(size.to_bytes(LENGTH_PREFIX_SIZE, "big"))

            # Send the compressed tensor data
            conn.sendall(compressed)
        except Exception as e:
            raise NetworkError(f"Failed to send tensor result: {e}")
