"""
Tensor compression utilities for network transmission in split computing.

This module provides PURE COMPRESSION ONLY for neural network tensors
to optimize network transmission in distributed computation environments.
Compression is completely separate from encryption.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
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
    """Handles PURE LOSSLESS COMPRESSION for distributed neural network computation.
    
    This class does ONLY compression/decompression:
    - Takes any data (encrypted or not)
    - Serializes with pickle
    - Compresses with blosc2
    - Reverses the process on decompression
    
    NO encryption logic, NO tensor shape analysis, NO model parameter extraction.
    Encryption should be handled separately BEFORE compression.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize tensor compression.

        Args:
            config: Compression parameters (clevel, filter, codec)
        """
        self.config = CompressionConfig(
            clevel=config.get("clevel", 3),
            filter=config.get("filter", "SHUFFLE"),
            codec=config.get("codec", "ZSTD"),
        )

        # Convert string filter names to Blosc2 constants
        filter_mapping = {
            "NOSHUFFLE": blosc2.Filter.NOFILTER,  # Map NOSHUFFLE to NOFILTER
            "SHUFFLE": blosc2.Filter.SHUFFLE,
            "BITSHUFFLE": blosc2.Filter.BITSHUFFLE,
        }
        self._filter = filter_mapping.get(self.config.filter, blosc2.Filter.SHUFFLE)

        # Convert string codec names to Blosc2 constants
        codec_mapping = {
            "ZSTD": blosc2.Codec.ZSTD,
            "LZ4": blosc2.Codec.LZ4,
            "BLOSCLZ": blosc2.Codec.BLOSCLZ,
        }
        self._codec = codec_mapping.get(self.config.codec, blosc2.Codec.ZSTD)

    def compress_data(self, data: Any) -> Tuple[bytes, int]:
        """Compress any data for network transmission.

        PURE COMPRESSION ONLY:
        1. Serialize the data with pickle (whatever it is - encrypted or not)
        2. Compress the serialized bytes with Blosc2
        3. Return compressed bytes and length

        Args:
            data: Any data to compress (encrypted tensors, plain tensors, etc.)

        Returns:
            Tuple of (compressed_bytes, compressed_length)
        """
        try:
            logger.debug(f"📦 Compressing data of type: {type(data)}")
            
            # STEP 1: Serialize (whatever the data is)
            serialized_data = pickle.dumps(data, protocol=HIGHEST_PROTOCOL)
            logger.debug(f"📦 Serialized data: {len(serialized_data)} bytes")

            # STEP 2: Compress 
            compressed_data = blosc2.compress(
                serialized_data,
                clevel=self.config.clevel,
                filter=self._filter,
                codec=self._codec,
            )
            logger.debug(f"📦 Compressed data: {len(compressed_data)} bytes (ratio: {len(compressed_data)/len(serialized_data):.2f})")

            # Return compressed bytes and their length
            return compressed_data, len(compressed_data)
            
        except Exception as e:
            logger.error(f"Data compression failed: {e}")
            raise CompressionError(f"Failed to compress data: {e}")

    def decompress_data(self, data: bytes) -> Any:
        """Decompress data.

        PURE DECOMPRESSION ONLY:
        1. Decompress using Blosc2
        2. Deserialize using pickle
        3. Return the original data (encrypted or not)

        Args:
            data: Compressed bytes to decompress

        Returns:
            Original data (whatever it was before compression)
        """
        try:
            logger.debug(f"📦 Decompressing data: {len(data)} bytes")
            
            # STEP 1: Decompress using Blosc2
            decompressed_data = blosc2.decompress(data)
            logger.debug(f"📦 Decompressed data: {len(decompressed_data)} bytes")

            # STEP 2: Deserialize using pickle
            original_data = pickle.loads(decompressed_data)
            logger.debug(f"📦 Deserialized data of type: {type(original_data)}")

            return original_data
            
        except Exception as e:
            logger.error(f"Data decompression failed: {e}")
            raise DecompressionError(f"Failed to decompress data: {e}")

    @staticmethod
    def _receive_chunk(conn: socket.socket, size: int) -> bytes:
        """Receive a specific sized chunk of data from a socket."""
        chunk = conn.recv(size)
        if not chunk:
            raise NetworkError("Socket connection broken during transmission")
        return chunk

    def receive_full_message(self, conn: socket.socket, expected_length: int) -> bytes:
        """Receive complete data of specified length from network connection."""
        if expected_length <= CHUNK_SIZE:
            return self._receive_chunk(conn, expected_length)

        # Allocate space for the complete data
        data_chunks = bytearray(expected_length)
        bytes_received = 0

        # Receive data in chunks until complete
        while bytes_received < expected_length:
            remaining = expected_length - bytes_received
            chunk_size = min(remaining, CHUNK_SIZE)

            try:
                chunk = self._receive_chunk(conn, chunk_size)
                data_chunks[bytes_received : bytes_received + len(chunk)] = chunk
                bytes_received += len(chunk)
            except Exception as e:
                logger.error(f"Error receiving data chunk: {e}")
                raise NetworkError(f"Failed to receive complete message: {e}")

        logger.debug(f"📦 Received complete message: {len(data_chunks)} bytes")
        return bytes(data_chunks)

    def receive_data(self, conn: socket.socket) -> Optional[Dict[str, Any]]:
        """Receive and decompress data from network connection."""
        try:
            # Read the length prefix
            length_bytes = self.receive_full_message(conn, LENGTH_PREFIX_SIZE)
            expected_length = int.from_bytes(length_bytes, byteorder="big")
            logger.debug(f"📦 Expecting message of length: {expected_length}")

            if expected_length <= 0:
                logger.warning("Received invalid message length")
                return None

            # Receive the compressed data
            compressed_data = self.receive_full_message(conn, expected_length)
            
            # Decompress and return
            return self.decompress_data(compressed_data)
            
        except Exception as e:
            logger.error(f"Error receiving data: {e}")
            return None

    def send_result(self, conn: socket.socket, result: Any) -> None:
        """Compress and send result over network connection."""
        try:
            # Compress the result
            compressed_data, compressed_length = self.compress_data(result)
            
            # Send length prefix
            length_bytes = compressed_length.to_bytes(LENGTH_PREFIX_SIZE, byteorder="big")
            conn.sendall(length_bytes)
            
            # Send compressed data
            conn.sendall(compressed_data)
            
            logger.debug(f"📦 Sent compressed result: {compressed_length} bytes")
            
        except Exception as e:
            logger.error(f"Error sending result: {e}")
            raise NetworkError(f"Failed to send result: {e}")
