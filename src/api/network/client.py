"""
Network client for split computing implementation.

This module provides client-side functionality for establishing connections
with a server and handling tensor transmission in a split computing architecture.
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
    SOCKET_TIMEOUT,
)
from .encryption import TensorEncryption
from .compression import DataCompression

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


class SplitComputeClient:
    """
    Handles communication between a client device and the split compute server.

    This class manages the network connection and tensor data transmission
    for distributed neural network processing.
    """

    def __init__(self, config: Dict[str, Any], encryption: Optional[TensorEncryption] = None) -> None:
        """
        Initialize the client with configuration and optional encryption.

        Args:
            config: Configuration dictionary containing network settings
            encryption: Optional TensorEncryption instance for secure communications
        """
        self.config = config
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.host: Optional[str] = None
        self.port: Optional[int] = None
        self.encryption = encryption

        # Initialize compression with encryption if provided
        compression_config = config.get("compression", {})
        encryption_mode = config.get("encryption", {}).get("mode", "transmission") if encryption else "transmission"
        self.compressor = DataCompression(compression_config, encryption=encryption, encryption_mode=encryption_mode)

        if encryption:
            logger.info(f"Client initialized with {encryption.mode} encryption")
        else:
            logger.info("Client initialized without encryption")

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
                
                # Get existing encryption config or create empty dict
                existing_encryption = config_with_context.get("encryption", {})
                
                # Add encryption context to the existing config for server synchronization
                encryption_config = existing_encryption.copy()
                encryption_config.update({
                    "context": self.encryption.serialize_context(),
                    "mode": self.encryption.mode,
                    "degree": self.encryption.degree,
                    "scale": self.encryption.scale
                })
                
                config_with_context["encryption"] = encryption_config
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
            
            # Wait for server ready signal (indicates server finished initialization)
            logger.info("Waiting for server to complete initialization...")
            
            # Set a longer timeout for initialization (encryption + model loading can be slow)
            original_timeout = self.socket.gettimeout()
            self.socket.settimeout(120)  # 2 minutes for server initialization
            
            try:
                ready_signal = self.socket.recv(5)  # "READY" is 5 bytes
                if ready_signal != b"READY":
                    logger.error(
                        f"Server ready signal failed: expected b'READY', got {ready_signal!r}"
                    )
                    self.close()
                    return False
                
                logger.info("Server is ready for tensor processing")
            except socket.timeout:
                logger.error("Timeout waiting for server initialization (2 minutes)")
                self.close()
                return False
            finally:
                # Restore original timeout
                self.socket.settimeout(original_timeout)
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
    config: Dict[str, Any],
    host: str,
    port: int = DEFAULT_PORT,
    encryption_context: Optional[bytes] = None,
    encryption_password: Optional[str] = None,
    encryption_key_file: Optional[str] = None,
    encryption: Optional[TensorEncryption] = None,
    **kwargs
) -> SplitComputeClient:
    """
    Factory function to create a properly configured network client.

    Args:
        config: Configuration dictionary for the client
        host: Server hostname or IP address
        port: Server port number
        encryption_context: Optional serialized encryption context
        encryption_password: Optional password for encryption
        encryption_key_file: Optional path to encryption key file
        encryption: Optional pre-configured TensorEncryption instance
        **kwargs: Additional keyword arguments

    Returns:
        Configured SplitComputeClient instance
    """
    # Initialize encryption if parameters are provided
    if encryption is None and (encryption_context or encryption_password or encryption_key_file):
        try:
            # Determine encryption mode from config
            encryption_config = config.get("encryption", {})
            encryption_mode = encryption_config.get("mode", "transmission")
            degree = encryption_config.get("degree", 8192)
            scale = encryption_config.get("scale", 26)
            
            if encryption_context:
                logger.info("Creating encryption from serialized context")
                # Note: Context-based encryption needs to be implemented
                logger.warning("Context-based encryption not yet implemented with new TensorEncryption")
                encryption = TensorEncryption(mode=encryption_mode, degree=degree, scale=scale)
            elif encryption_password:
                logger.info("Creating encryption from password")
                encryption = TensorEncryption.from_password(
                    password=encryption_password,
                    mode=encryption_mode,
                    degree=degree,
                    scale=scale
                )
            elif encryption_key_file:
                logger.info(f"Creating encryption from key file: {encryption_key_file}")
                logger.warning("Key file-based encryption not yet implemented with new TensorEncryption")
                encryption = TensorEncryption(mode=encryption_mode, degree=degree, scale=scale)
        except Exception as e:
            logger.error(f"Failed to initialize client encryption: {e}")
            encryption = None

    client = SplitComputeClient(config, encryption=encryption)
    client.host = host
    client.port = port

    return client


def create_client_with_encryption(
    config: Dict[str, Any],
    host: str,
    port: int = DEFAULT_PORT,
    encryption: Optional[TensorEncryption] = None,
    **kwargs
) -> SplitComputeClient:
    """
    Create a network client with encryption support.

    Args:
        config: Configuration dictionary for the client
        host: Server hostname or IP address
        port: Server port number
        encryption: Optional pre-configured TensorEncryption instance
        **kwargs: Additional keyword arguments

    Returns:
        Configured SplitComputeClient instance with encryption
    """
    return create_network_client(
        config=config,
        host=host,
        port=port,
        encryption=encryption,
        **kwargs
    )
