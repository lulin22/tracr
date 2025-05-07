"""
Network client implementation for split computing tensor transmission.

This module implements client-side networking for distributed neural network computation,
handling connections to server-side processing nodes and tensor data transfer.
"""

import pickle
import socket
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .protocols import (
    LENGTH_PREFIX_SIZE,
    BUFFER_SIZE,
    ACK_MESSAGE,
    HIGHEST_PROTOCOL,
    DEFAULT_COMPRESSION_SETTINGS,
    DEFAULT_PORT,
)
from .encryption import HomomorphicEncryption, create_encryption

try:
    import blosc2

    BLOSC2_AVAILABLE = True
    logger = logging.getLogger("split_computing_logger")
    logger.info("Using blosc2 compression (codec: ZSTD, filter: SHUFFLE, level: 3)")
except ImportError:
    import zlib

    BLOSC2_AVAILABLE = False
    logging.getLogger("split_computing_logger").warning(
        "blosc2 not available, falling back to zlib (slower compression)"
    )

logger = logging.getLogger("split_computing_logger")


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for network connection and tensor transmission."""

    config: Dict[str, Any]  # Experiment configuration including compression settings
    host: str  # Server host address for tensor transmission
    port: int  # Server port for tensor transmission


class NetworkError(Exception):
    """Base exception for network-related errors in tensor transmission."""

    pass


class CompressionError(Exception):
    """Base exception for tensor compression-related errors."""

    pass


class DecompressionError(CompressionError):
    """Exception raised when tensor decompression fails."""

    pass


class DataCompression:
    """Handles tensor compression and decompression to minimize network transmission overhead."""

    def __init__(self, config: Dict[str, Any], encryption: Optional[HomomorphicEncryption] = None) -> None:
        """
        Initialize compression and encryption engine with configuration.
        
        Args:
            config: Compression configuration
            encryption: Optional encryption module for secure tensor transmission
        """
        self.config = config
        self.encryption = encryption
        self.encryption_enabled = encryption is not None
        
        if self.encryption_enabled:
            logger.info("Homomorphic encryption enabled for tensor transmission")
            
        if BLOSC2_AVAILABLE:
            # Configure optimal tensor compression parameters
            self._filter = blosc2.Filter.SHUFFLE
            self._codec = blosc2.Codec.ZSTD

            # Override with user-specified filter if valid
            if "filter" in self.config:
                try:
                    self._filter = blosc2.Filter[self.config["filter"]]
                except (KeyError, AttributeError):
                    logger.warning(
                        f"Invalid blosc2 filter: {self.config['filter']}, using SHUFFLE"
                    )

            # Override with user-specified codec if valid
            if "codec" in self.config:
                try:
                    self._codec = blosc2.Codec[self.config["codec"]]
                except (KeyError, AttributeError):
                    logger.warning(
                        f"Invalid blosc2 codec: {self.config['codec']}, using ZSTD"
                    )

            if "clevel" not in self.config:
                self.config["clevel"] = 3  # Default compression level

    def compress_data(self, data: Any) -> Tuple[bytes, int]:
        """
        Compress tensor data for efficient network transmission.

        === TENSOR SHARING PIPELINE - STAGE 1: COMPRESSION ===
        Serializes and compresses tensors before network transmission to reduce bandwidth
        requirements. This method is critical for efficient tensor sharing between devices.
        """
        try:
            # First serialize the tensor data structure using pickle
            serialized_data = pickle.dumps(data, protocol=HIGHEST_PROTOCOL)

            # Apply compression algorithm based on available libraries
            if BLOSC2_AVAILABLE:
                compressed_data = blosc2.compress(
                    serialized_data,
                    clevel=self.config["clevel"],
                    filter=self._filter,
                    codec=self._codec,
                )
            else:
                compressed_data = zlib.compress(
                    serialized_data, level=self.config["clevel"]
                )

            # Apply encryption if enabled
            if self.encryption_enabled:
                try:
                    # Use homomorphic encryption for secure transmission
                    encrypted_data, metadata = self.encryption.encrypt(compressed_data)
                    
                    # Serialize metadata for transmission
                    metadata_bytes = pickle.dumps(metadata, protocol=HIGHEST_PROTOCOL)
                    
                    # Format data as: [metadata_length (4 bytes)][metadata][encrypted_data]
                    metadata_len = len(metadata_bytes)
                    final_data = metadata_len.to_bytes(4, byteorder='big') + metadata_bytes + encrypted_data
                    
                    return final_data, len(final_data)
                except Exception as e:
                    logger.error(f"Tensor encryption failed: {e}")
                    raise CompressionError(f"Failed to encrypt tensor data: {e}")
            
            # Return compressed data without encryption
            return compressed_data, len(compressed_data)
        except Exception as e:
            logger.error(f"Tensor compression failed: {e}")
            raise CompressionError(f"Failed to compress tensor data: {e}")

    def decompress_data(self, compressed_data: bytes) -> Any:
        """
        Decompress received tensor data from network transmission.

        === TENSOR SHARING PIPELINE - STAGE 3: DECOMPRESSION ===
        Decompresses and deserializes tensor data received from the network,
        recovering the original tensor structure for computational processing.
        """
        try:
            # Handle encrypted data if encryption is enabled
            if self.encryption_enabled:
                # Extract metadata length (first 4 bytes)
                metadata_len = int.from_bytes(compressed_data[:4], byteorder='big')
                
                # Extract metadata
                metadata_bytes = compressed_data[4:4+metadata_len]
                metadata = pickle.loads(metadata_bytes)
                
                # Extract encrypted data
                encrypted_data = compressed_data[4+metadata_len:]
                
                # Decrypt the data
                decrypted_data = self.encryption.decrypt(encrypted_data, metadata)
                
                # If the result is already a tensor, return it
                if not isinstance(decrypted_data, bytes):
                    return decrypted_data
                
                # Otherwise, decompress and deserialize
                decompressed_data = decrypted_data
            else:
                # Apply decompression without decryption
                decompressed_data = compressed_data
            
            # Apply decompression algorithm based on available libraries
            if BLOSC2_AVAILABLE:
                decompressed = blosc2.decompress(decompressed_data)
            else:
                decompressed = zlib.decompress(decompressed_data)

            # Deserialize to recover original tensor
            return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Tensor decompression failed: {e}")
            raise DecompressionError(f"Failed to decompress tensor data: {e}")

    def receive_full_message(self, conn: socket.socket, expected_length: int) -> bytes:
        """
        Receive a complete tensor data message of expected length from a socket.

        === TENSOR SHARING PIPELINE - STAGE 2: DATA RECEPTION ===
        Handles fragmentation of large tensors by receiving in chunks until the complete
        tensor is received or a connection error occurs.
        """
        data = bytearray()
        received = 0

        while received < expected_length:
            try:
                remaining = expected_length - received
                chunk = conn.recv(min(BUFFER_SIZE, remaining))

                if not chunk:
                    logger.error(
                        f"Connection closed while receiving tensor data ({received}/{expected_length} bytes received)"
                    )
                    raise NetworkError("Connection closed while receiving tensor data")

                data.extend(chunk)
                received += len(chunk)

            except socket.timeout:
                logger.error("Socket timed out while receiving tensor data")
                raise NetworkError("Socket timed out while receiving tensor data")
            except ConnectionError as e:
                logger.error(f"Connection error while receiving tensor data: {e}")
                raise NetworkError(f"Connection error: {e}")
            except Exception as e:
                logger.error(f"Error receiving tensor data: {e}")
                raise NetworkError(f"Error receiving tensor data: {e}")

        return bytes(data)


class SplitComputeClient:
    """Manages client-side network operations for distributed tensor computation."""

    def __init__(
        self, 
        network_config: NetworkConfig, 
        encryption_context: Optional[bytes] = None,
        encryption_password: Optional[str] = None,
        poly_modulus_degree: int = 8192,
        bit_scale: int = 26,
        encryption: Optional[HomomorphicEncryption] = None
    ) -> None:
        """
        Initialize client with network configuration and optional encryption.
        
        Args:
            network_config: Network configuration settings
            encryption_context: Optional serialized TenSEAL context for homomorphic encryption
            encryption_password: Optional password to derive encryption parameters
            poly_modulus_degree: Polynomial modulus degree for homomorphic encryption (power of 2)
            bit_scale: Bit scale for encoding precision in homomorphic encryption
            encryption: Optional pre-configured HomomorphicEncryption instance
        """
        self.config = network_config.config
        self.host = network_config.host
        self.port = network_config.port
        self.socket = None
        self.connected = False

        # Setup encryption if parameters are provided
        self.encryption = encryption
        
        # If encryption instance is not directly provided, try to create one from parameters
        if self.encryption is None:
            if encryption_context is not None:
                # Create encryption from serialized context
                try:
                    self.encryption = HomomorphicEncryption.from_serialized_context(encryption_context)
                    logger.info("Initialized homomorphic encryption from provided context")
                except Exception as e:
                    logger.error(f"Failed to initialize encryption from context: {e}")
                    raise
            elif encryption_password is not None:
                # Create encryption from password
                try:
                    self.encryption = HomomorphicEncryption.from_password(
                        encryption_password, 
                        poly_modulus_degree=poly_modulus_degree
                    )
                    logger.info("Initialized homomorphic encryption from password")
                except Exception as e:
                    logger.error(f"Failed to initialize encryption from password: {e}")
                    raise
        elif encryption_context is not None or encryption_password is not None:
            logger.warning("Both direct encryption instance and parameters provided; using direct instance")
        
        # Initialize compression with optional encryption
        compression_config = self.config.get(
            "compression", {"clevel": 3, "filter": "SHUFFLE", "codec": "ZSTD"}
        )
        self.compressor = DataCompression(
            compression_config,
            encryption=self.encryption
        )

    def connect(self) -> bool:
        """
        Establish connection to computation server and send initial configuration.

        === TENSOR SHARING SETUP PHASE ===
        Creates the network channel through which tensors will be shared,
        and synchronizes configuration settings with the server.
        """
        if self.connected and self.socket:
            return True

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            logger.info(f"Connected to {self.host}:{self.port}")

            # If using encryption, add the serialized context to the config
            if self.encryption is not None:
                # Create a copy of the config to avoid modifying the original
                config_with_context = self.config.copy()
                # Add encryption context to config for server synchronization
                config_with_context["encryption"] = {
                    "context": self.encryption.serialize_context()
                }
                config_bytes = pickle.dumps(config_with_context, protocol=HIGHEST_PROTOCOL)
            else:
                # Serialize configuration without encryption
                config_bytes = pickle.dumps(self.config, protocol=HIGHEST_PROTOCOL)

            # Send length-prefixed configuration in one atomic operation
            # This ensures the server receives configuration parameters before any tensor data
            size_bytes = len(config_bytes).to_bytes(LENGTH_PREFIX_SIZE, "big")
            self.socket.sendall(size_bytes + config_bytes)

            # Wait for server acknowledgment before proceeding
            ack = self.socket.recv(len(ACK_MESSAGE))
            if ack != ACK_MESSAGE:
                logger.error(
                    f"Server acknowledgment failed: expected {ACK_MESSAGE!r}, got {ack!r}"
                )
                self.close()
                return False

            logger.info("Server acknowledged configuration")
            self.connected = True
            return True

        except Exception as e:
            logger.error(f"Connection setup failed: {e}")
            if self.socket:
                self.close()
            return False

    def process_split_computation(
        self, split_index: int, intermediate_output: bytes
    ) -> Tuple[Any, float]:
        """
        Send intermediate tensor to the server for continued computation.

        === TENSOR SHARING - CLIENT SIDE ===
        This is the core tensor sharing method that:
        1. Sends compressed intermediate tensor data to the server
        2. Waits for the server to process the tensor
        3. Receives and decompresses the computed result tensor

        Args:
            split_index: Layer index where the model was split
            intermediate_output: Compressed intermediate tensor data

        Returns:
            Tuple of (processed_result, server_time)
        """
        if not self.connected or not self.socket:
            if not self.connect():
                raise NetworkError("Failed to connect to server")

        try:
            # Prepare header containing split point and tensor size information
            # This informs the server which model layer to resume computation from
            header = split_index.to_bytes(LENGTH_PREFIX_SIZE, "big") + len(
                intermediate_output
            ).to_bytes(LENGTH_PREFIX_SIZE, "big")

            # Send the header and compressed tensor in sequence
            self.socket.sendall(header)
            self.socket.sendall(intermediate_output)
            logger.debug(
                f"Sent {len(intermediate_output)} bytes for split layer {split_index}"
            )

            # Receive expected result size to properly handle fragmentation
            result_size_bytes = self.socket.recv(LENGTH_PREFIX_SIZE)
            if not result_size_bytes or len(result_size_bytes) != LENGTH_PREFIX_SIZE:
                raise NetworkError(
                    "Connection closed by server while reading result size"
                )

            result_size = int.from_bytes(result_size_bytes, "big")
            logger.debug(f"Server will send {result_size} bytes of tensor result data")

            # Receive server processing time metric
            server_time_bytes = self.socket.recv(LENGTH_PREFIX_SIZE)
            if not server_time_bytes or len(server_time_bytes) != LENGTH_PREFIX_SIZE:
                raise NetworkError(
                    "Connection closed by server while reading server time"
                )

            try:
                server_time = float(server_time_bytes.strip().decode())
                logger.debug(f"Server tensor processing time: {server_time}s")
            except ValueError:
                logger.error(f"Invalid server time received: {server_time_bytes!r}")
                server_time = 0.0

            # Receive the compressed result tensor data
            response_data = self.compressor.receive_full_message(
                conn=self.socket, expected_length=result_size
            )
            logger.debug(
                f"Received {len(response_data)} bytes of compressed result tensor"
            )

            # Decompress the tensor result
            processed_result = self.compressor.decompress_data(response_data)

            return processed_result, server_time

        except Exception as e:
            logger.error(f"Split tensor computation failed: {e}")
            self.close()
            raise NetworkError(f"Failed to process split tensor computation: {e}")

    def close(self) -> None:
        """Close the socket connection used for tensor transmission."""
        if self.socket:
            try:
                self.socket.close()
                logger.info("Tensor transmission connection closed")
            except Exception as e:
                logger.warning(f"Error closing tensor transmission socket: {e}")
            finally:
                self.socket = None
                self.connected = False


def create_network_client(
    config: Optional[Dict[str, Any]] = None,
    host: str = "localhost",
    port: int = DEFAULT_PORT,
    encryption_password: Optional[str] = None,
    poly_modulus_degree: int = 8192,
    bit_scale: int = 26,
    encryption: Optional[HomomorphicEncryption] = None,
) -> SplitComputeClient:
    """
    Create a network client for tensor sharing in split computing.

    This factory function creates a properly configured client that can
    transmit intermediate tensors to the server component of the split
    computing architecture.
    
    Args:
        config: Configuration dictionary
        host: Server hostname or IP address
        port: Server port number
        encryption_password: Optional password for homomorphic encryption
        poly_modulus_degree: Polynomial modulus degree for homomorphic encryption
        bit_scale: Bit scale for encoding precision
        encryption: Optional pre-configured HomomorphicEncryption instance
        
    Returns:
        Configured SplitComputeClient
    """
    if config is None:
        config = {}

    # Ensure compression configuration for efficient tensor transmission
    if "compression" not in config:
        config["compression"] = DEFAULT_COMPRESSION_SETTINGS

    network_config = NetworkConfig(config=config, host=host, port=port)
    
    # Create client with optional encryption
    client = SplitComputeClient(
        network_config,
        encryption_password=encryption_password if encryption is None else None,
        poly_modulus_degree=poly_modulus_degree,
        bit_scale=bit_scale,
        encryption=encryption
    )
    
    return client
