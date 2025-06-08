#!/usr/bin/env python
"""
Server-side implementation of the split computing architecture.

This module implements the server side of a split computing architecture.
It can be run in either networked mode (handling connections from clients) or local mode
(running experiments locally without network communication).
"""

import logging
import pickle
import socket
import sys
import time
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, Generator
import os

import torch

# Add project root to path so we can import from src module
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.api import ( # noqa: E402
    DataCompression,
    DeviceManager,
    ExperimentManager,
    DeviceType,
    start_logging_server,
    shutdown_logging_server,
    DataCompression, # noqa: F811
    read_yaml_file,
)
from src.api.network.protocols import ( # noqa: E402
    LENGTH_PREFIX_SIZE,
    ACK_MESSAGE,
    SERVER_COMPRESSION_SETTINGS,
    SERVER_LISTEN_TIMEOUT,
    SOCKET_TIMEOUT,
    DEFAULT_PORT,
)

DEFAULT_CONFIG: Dict[str, Any] = {
    "logging": {"log_file": "logs/server.log", "log_level": "INFO"}
}

# Start logging server
logging_server = start_logging_server(
    device=DeviceType.SERVER, 
    config=DEFAULT_CONFIG,
    find_free_port=True  # Try to find a free port if the default is in use
)
logger = logging.getLogger("split_computing_logger")


def get_device(requested_device: str = "cuda") -> str:
    """Determine the appropriate device based on availability and request."""
    requested_device = requested_device.lower()

    if requested_device == "cpu":
        logger.info("CPU device explicitly requested")
        return "cpu"

    if requested_device == "cuda" and torch.cuda.is_available():
        logger.info("CUDA is available and will be used")
        return "cuda"

    # Check for MPS (Apple Silicon GPUs)
    if (
        requested_device == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        logger.info("MPS (Apple Silicon GPU) is available and will be used")
        return "mps"

    # If we're here, requested GPU is not available - try alternatives
    if requested_device in ("cuda", "gpu", "mps"):
        # If any GPU was requested, try all available options in priority order
        if torch.cuda.is_available():
            logger.info(
                f"{requested_device.upper()} requested but not available, using CUDA instead"
            )
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info(
                f"{requested_device.upper()} requested but not available, using MPS (Apple Silicon GPU) instead"
            )
            return "mps"

        logger.warning(
            f"{requested_device.upper()} requested but no GPU available, falling back to CPU"
        )
        return "cpu"

    # For any other requested device, fall back to CPU
    logger.warning(
        f"Requested device '{requested_device}' not recognized, falling back to CPU"
    )
    return "cpu"


@dataclass
class ServerMetrics:
    """Container for metrics collected during server operation."""

    total_requests: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0

    def update(self, processing_time: float) -> None:
        """Update metrics with a new processing time measurement."""
        self.total_requests += 1
        self.total_processing_time += processing_time
        self.avg_processing_time = self.total_processing_time / self.total_requests


class Server:
    """
    Handles server operations for managing connections and processing data.

    This class implements both networked and local modes:
    - Networked mode: listens for client connections and processes data sent by clients
    - Local mode: runs experiments locally using the provided configuration
    """

    def __init__(
        self, 
        local_mode: bool = False, 
        config_path: Optional[str] = None,
        enable_encryption: bool = False,
        encryption_key_file: Optional[str] = None,
        encryption_password: Optional[str] = None,
        encryption_degree: int = 8192,
        encryption_scale: int = 26
    ) -> None:
        """
        Initialize the Server with specified mode and configuration.
        
        Args:
            local_mode: Whether to run in local mode (True) or networked mode (False)
            config_path: Path to configuration file
            enable_encryption: Whether to enable encryption
            encryption_key_file: Path to encryption key file
            encryption_password: Password for encryption
            encryption_degree: Polynomial modulus degree for encryption
            encryption_scale: Bit scale for encryption precision
        """
        self.device_manager = DeviceManager()
        self.experiment_manager: Optional[ExperimentManager] = None
        self.server_socket: Optional[socket.socket] = None
        self.local_mode = local_mode
        self.config_path = config_path
        self.metrics = ServerMetrics()
        self.compress_data: Optional[DataCompression] = None
        
        # Store encryption settings
        self.enable_encryption = enable_encryption
        self.encryption_key_file = encryption_key_file
        self.encryption_password = encryption_password
        self.encryption_degree = encryption_degree
        self.encryption_scale = encryption_scale

        self._load_config_and_setup_device()
        # Setup compression if in networked mode
        if not local_mode:
            self._setup_compression()
            logger.debug("Server initialized in network mode")
        else:
            logger.debug("Server initialized in local mode")

    def _load_config_and_setup_device(self) -> None:
        """Load configuration and set up device."""
        if not self.config_path:
            return

        self.config = read_yaml_file(self.config_path)
        requested_device = self.config.get("default", {}).get("device", "cuda")
        self.config["default"]["device"] = get_device(requested_device)
        
        # Check for encryption settings in the config file if not provided via command line
        if not self.enable_encryption:
            config_encryption = self.config.get("encryption", {})
            self.enable_encryption = config_encryption.get("enabled", False)
            
            # Only use config encryption settings if not provided via command line
            if self.enable_encryption:
                if not self.encryption_key_file:
                    self.encryption_key_file = config_encryption.get("key_file")
                
                if not self.encryption_password:
                    self.encryption_password = config_encryption.get("password")
                    
                if self.encryption_degree == 8192:  # Using default value
                    self.encryption_degree = config_encryption.get("degree", 8192)
                    
                if self.encryption_scale == 26:  # Using default value
                    self.encryption_scale = config_encryption.get("scale", 26)
                    
                logger.info("Encryption settings loaded from config file")

    def _setup_compression(self) -> None:
        """Initialize compression with minimal settings for optimal performance."""
        
        # Initialize encryption if enabled
        self.encryption = None
        
        if self.enable_encryption:
            try:
                from src.api.network.encryption import TensorEncryption
                
                # Determine encryption mode from config or default to transmission
                encryption_mode = "transmission"  # Default mode
                
                if hasattr(self, 'config') and self.config:
                    encryption_config = self.config.get("encryption", {})
                    encryption_mode = encryption_config.get("mode", "transmission")
                
                logger.info(f"Initializing server encryption in {encryption_mode} mode...")
                
                if self.encryption_key_file:
                    # Use specified key file from command line
                    logger.info(f"Using encryption key file: {self.encryption_key_file}")
                    logger.warning("Key file-based encryption not yet implemented with new TensorEncryption")
                    # For now, generate a new encryption instance
                    self.encryption = TensorEncryption(
                        mode=encryption_mode,
                        degree=self.encryption_degree,
                        scale=self.encryption_scale
                    )
                elif self.encryption_password:
                    # Use specified password from command line
                    logger.info("Using encryption with password")
                    self.encryption = TensorEncryption.from_password(
                        password=self.encryption_password,
                        mode=encryption_mode,
                        degree=self.encryption_degree,
                        scale=self.encryption_scale
                    )
                else:
                    # Generate new encryption instance if no specific method provided
                    logger.info("Generating new encryption context")
                    self.encryption = TensorEncryption(
                        mode=encryption_mode,
                        degree=self.encryption_degree,
                        scale=self.encryption_scale
                    )
                
                logger.info(f"Server encryption initialized successfully with mode={encryption_mode}")
                    
            except Exception as e:
                logger.error(f"Failed to initialize encryption: {e}")
                self.encryption = None
        
        # Fallback to environment variable if not explicitly enabled
        elif os.environ.get("SPLIT_COMPUTE_ENCRYPTION_KEY"):
            try:
                from src.api.network.encryption import TensorEncryption
                
                encryption_key_file = os.environ.get("SPLIT_COMPUTE_ENCRYPTION_KEY")
                if os.path.exists(encryption_key_file):
                    logger.info(f"Using encryption key from environment: {encryption_key_file}")
                    logger.warning("Key file-based encryption not yet implemented with new TensorEncryption")
                    # For now, generate a new encryption instance
                    self.encryption = TensorEncryption(
                        mode="transmission",  # Default for environment-based setup
                        degree=self.encryption_degree,
                        scale=self.encryption_scale
                    )
            except Exception as e:
                logger.error(f"Failed to initialize encryption from environment: {e}")
                self.encryption = None
        
        # Initialize compression (always needed)
        self.compress_data = DataCompression(
            SERVER_COMPRESSION_SETTINGS, 
            encryption=self.encryption
        )
        
        if self.encryption:
            logger.info(f"Homomorphic encryption enabled for secure tensor processing in {self.encryption.mode} mode")
        else:
            logger.info("Compression initialized without encryption")
        
        logger.debug("Initialized compression with minimal settings")

    def start(self) -> None:
        """Start the server in either networked or local mode."""
        if self.local_mode:
            self._run_local_experiment()
        else:
            self._run_networked_server()

    def _run_local_experiment(self) -> None:
        """Run experiment locally on the server."""
        if not self.config_path:
            logger.error("Config path required for local mode")
            return

        try:
            logger.info("Starting local experiment...")
            self._setup_and_run_local_experiment()
            logger.info("Local experiment completed successfully")
        except Exception as e:
            logger.error(f"Error running local experiment: {e}", exc_info=True)

    def _setup_and_run_local_experiment(self) -> None:
        """Set up and run a local experiment based on configuration."""
        from src.experiment_design.datasets.core.loaders import DatasetRegistry
        import torch.utils.data

        config = read_yaml_file(self.config_path)
        self.experiment_manager = ExperimentManager(config, force_local=True)
        experiment = self.experiment_manager.setup_experiment()

        # Set up data loader
        dataset_config = config.get("dataset", {})
        dataloader_config = config.get("dataloader", {})

        # Get the appropriate collate function if specified
        collate_fn = self._get_collate_function(dataloader_config)

        # Get dataset name - required parameter
        dataset_name = dataset_config.get("name")
        if not dataset_name:
            logger.error("Dataset name not specified in config (required 'name' field)")
            return

        # Create a copy of the dataset config for loading
        complete_config = dataset_config.copy()

        # Add transform from dataloader config if not already specified
        if "transform" not in complete_config and "transform" in dataloader_config:
            complete_config["transform"] = dataloader_config.get("transform")

        # Load dataset using registry
        try:
            # First register the dataset if needed
            if DatasetRegistry.get_metadata(dataset_name) is None:
                logger.info(f"Registering dataset '{dataset_name}'")
                DatasetRegistry.register_dataset(dataset_name)

            # Now load the dataset
            dataset = DatasetRegistry.load(complete_config)
            logger.info(f"Loaded dataset '{dataset_name}' successfully")
        except Exception as e:
            logger.error(f"Failed to load dataset '{dataset_name}': {e}")
            raise  # Re-raise to ensure the error is properly handled

        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataloader_config.get("batch_size"),
            shuffle=dataloader_config.get("shuffle"),
            num_workers=dataloader_config.get("num_workers"),
            collate_fn=collate_fn,
        )

        # Attach data loader to experiment and run
        experiment.data_loader = data_loader
        experiment.run()

    def _get_collate_function(self, dataloader_config: Dict[str, Any]) -> Optional[Any]:
        """
        Get the collate function specified in the configuration.

        Collate functions customize how individual data samples are combined
        into batches for model processing.
        """
        if not dataloader_config.get("collate_fn"):
            return None

        try:
            from src.experiment_design.datasets.core.collate_fns import CollateRegistry

            collate_fn_name = dataloader_config["collate_fn"]
            collate_fn = CollateRegistry.get(collate_fn_name)

            if not collate_fn:
                logger.warning(
                    f"Collate function '{collate_fn_name}' not found in registry. "
                    "Using default collation."
                )
                return None

            logger.debug(f"Using registered collate function: {collate_fn_name}")
            return collate_fn
        except ImportError as e:
            logger.warning(
                f"Failed to import collate functions: {e}. Using default collation."
            )
            return None
        except KeyError:
            logger.warning(
                f"Collate function '{dataloader_config['collate_fn']}' not found. "
                "Using default collation."
            )
            return None

    def _run_networked_server(self) -> None:
        """Run server in networked mode, accepting client connections."""
        # Get server device configuration
        server_device = self.device_manager.get_device_by_type("SERVER")
        if not server_device:
            logger.error("No SERVER device configured. Cannot start server.")
            return

        if not server_device.is_reachable():
            logger.error("SERVER device is not reachable. Check network connection.")
            return

        # Use experiment port for network communication
        port = server_device.get_port()
        if port is None:
            logger.info(
                f"No port configured for SERVER device, using DEFAULT_PORT={DEFAULT_PORT}"
            )
            port = DEFAULT_PORT

        logger.info(f"Starting networked server on port {port}...")

        try:
            self._setup_socket(port)
            self._accept_connections()
        except KeyboardInterrupt:
            logger.info("Server shutdown requested...")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            self.cleanup()

    def _accept_connections(self) -> None:
        """
        Accept and handle client connections in a continuous loop.

        Uses socket timeout to allow for graceful shutdown on keyboard interrupt.
        """
        while True:
            try:
                conn, addr = self.server_socket.accept()
                # Set timeout on client socket for data operations
                conn.settimeout(SOCKET_TIMEOUT)
                logger.info(f"Connected by {addr}")
                self.handle_connection(conn)
            except socket.timeout:
                # Handle timeout, allow checking for keyboard interrupt
                continue
            except ConnectionError as e:
                logger.error(f"Connection error: {e}")
                continue

    def _setup_socket(self, port: int) -> None:
        """
        Set up server socket with proper error handling.

        Creates a socket that:
        - Allows address reuse (SO_REUSEADDR)
        - Has a timeout to enable graceful shutdown
        - Listens on all interfaces (empty host string)
        """
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Set a timeout to allow graceful shutdown on keyboard interrupt
            self.server_socket.settimeout(SERVER_LISTEN_TIMEOUT)
            self.server_socket.bind(("", port))
            self.server_socket.listen()
            logger.info(f"Server is listening on port {port} (all interfaces)")
        except Exception as e:
            logger.error(f"Failed to create server socket: {e}")
            raise

    def _receive_config(self, conn: socket.socket) -> dict:
        """
        Receive and parse configuration from client.

        Implements a length-prefixed protocol for receiving structured data:
        1. First 4 bytes indicate the total message length
        2. Remaining bytes contain the serialized configuration

        Returns:
            The deserialized configuration dictionary
        """
        try:
            # Read the length prefix (4 bytes)
            config_length_bytes = conn.recv(LENGTH_PREFIX_SIZE)
            if (
                not config_length_bytes
                or len(config_length_bytes) != LENGTH_PREFIX_SIZE
            ):
                logger.error("Failed to receive config length prefix")
                return {}

            config_length = int.from_bytes(config_length_bytes, "big")
            logger.debug(f"Expecting config data of length {config_length} bytes")

            if not self.compress_data:
                logger.error("Compression not initialized")
                return {}

            # Receive the raw config data (no compression for config)
            config_data = self.compress_data.receive_full_message(
                conn=conn, expected_length=config_length
            )

            if not config_data:
                logger.error("Failed to receive config data")
                return {}

            # Deserialize using pickle
            try:
                config = pickle.loads(config_data)
                logger.debug(f"Successfully received and parsed configuration")
                
                # Check for encryption context in configuration
                if "encryption" in config and isinstance(config["encryption"], dict):
                    encryption_config = config["encryption"]
                    if "context" in encryption_config:
                        try:
                            from src.api.network.encryption import TensorEncryption
                            serialized_context = encryption_config["context"]
                            
                            # Properly deserialize the client's encryption context
                            logger.info("Deserializing encryption context from client...")
                            
                            encryption_mode = encryption_config.get("mode", "transmission")
                            
                            # Use the from_serialized_context method to create compatible encryption
                            server_encryption = TensorEncryption.from_serialized_context(
                                serialized_context, 
                                mode=encryption_mode
                            )
                            
                            # Update the server's encryption for this session
                            self.encryption = server_encryption
                            logger.info(f"Successfully initialized server encryption from client context in {encryption_mode} mode")
                            
                        except Exception as e:
                            logger.error(f"Failed to initialize encryption from client context: {e}")
                            logger.warning("Continuing without encryption synchronization")
                            # Don't raise here - continue without encryption to allow connection to proceed
                
                return config
            except Exception as e:
                logger.error(f"Failed to deserialize config: {e}")
                return {}

        except Exception as e:
            logger.error(f"Error receiving config: {e}")
            return {}

    def _process_data(
        self,
        experiment: Any,
        output: torch.Tensor,
        original_size: Tuple[int, int],
        split_layer_index: int,
    ) -> Tuple[Any, float]:
        """
        Process received tensor data through the model and measure performance.

        This is the core split computation function that:
        1. Continues model execution from the specified split point
        2. Returns both the processed result and the time taken

        Args:
            experiment: The experiment object that will process the data
            output: The tensor output from the client
            original_size: Original size information
            split_layer_index: The index of the split layer

        Returns:
            Tuple of (processed_result, processing_time)
        """
        server_start_time = time.time()
        processed_result = experiment.process_data(
            {"input": (output, original_size), "split_layer": split_layer_index}
        )
        return processed_result, time.time() - server_start_time

    @contextmanager
    def _safe_connection(self, conn: socket.socket) -> Generator[None, None, None]:
        """
        Context manager for safely handling client connections.

        Ensures proper exception handling and connection cleanup regardless
        of how the connection processing terminates.
        """
        try:
            yield
        except Exception as e:
            logger.error(f"Error handling connection: {e}", exc_info=True)
        finally:
            try:
                conn.close()
            except Exception as e:
                logger.debug(f"Error closing connection: {e}")

    def handle_connection(self, conn: socket.socket) -> None:
        """
        Handle an individual client connection for split computing.

        The connection handling protocol follows these steps:
        1. Receive experiment configuration from client
        2. Initialize experiment based on received configuration
        3. Send acknowledgment to client
        4. Enter processing loop to handle tensor data
           - Receive intermediate tensors from client
           - Process tensors through the model from the split point
           - Send results back to client
        """
        with self._safe_connection(conn):
            # Receive configuration from the client
            config = self._receive_config(conn)
            if not config:
                logger.error("Failed to receive valid configuration from client")
                return

            # Update compression settings based on received config
            self._update_compression(config)

            # Initialize experiment based on received configuration
            try:
                self.experiment_manager = ExperimentManager(config)
                experiment = self.experiment_manager.setup_experiment()
                experiment.model.eval()
                
                # Pass encryption instance to experiment for homomorphic processing
                if self.encryption:
                    experiment.encryption = self.encryption
                    logger.info("Passed encryption instance to experiment for homomorphic operations")
                
                logger.info("Experiment initialized successfully with received config")
            except Exception as e:
                logger.error(f"Failed to initialize experiment: {e}")
                return

            # Cache torch.no_grad() context for inference
            no_grad_context = torch.no_grad()

            # Send acknowledgment to the client - must be exactly b"OK"
            conn.sendall(ACK_MESSAGE)
            logger.debug("Sent 'OK' acknowledgment to client")
            
            # Send READY signal after full initialization (encryption + model loading complete)
            conn.sendall(b"READY")
            logger.debug("Sent 'READY' signal to client - server fully initialized")

            # Process incoming data in a loop
            while True:
                try:
                    # Receive header - 8 bytes total (4 for split index, 4 for length)
                    header = conn.recv(LENGTH_PREFIX_SIZE * 2)
                    if not header or len(header) != LENGTH_PREFIX_SIZE * 2:
                        logger.info("Client disconnected or sent invalid header")
                        break

                    split_layer_index = int.from_bytes(
                        header[:LENGTH_PREFIX_SIZE], "big"
                    )
                    expected_length = int.from_bytes(header[LENGTH_PREFIX_SIZE:], "big")
                    logger.debug(
                        f"Received header: split_layer={split_layer_index}, data_length={expected_length}"
                    )

                    # Receive compressed data from client
                    if not self.compress_data:
                        logger.error("Compression not initialized")
                        break

                    compressed_data = self.compress_data.receive_full_message(
                        conn=conn, expected_length=expected_length
                    )

                    if not compressed_data:
                        logger.warning("Failed to receive compressed data from client")
                        break

                    logger.debug(
                        f"Received {len(compressed_data)} bytes of compressed data"
                    )

                    # Process the data
                    with no_grad_context:
                        # Decompress received data
                        output, original_size = self.compress_data.decompress_data(
                            data=compressed_data
                        )

                        # Process data using the experiment's model
                        processed_result, processing_time = self._process_data(
                            experiment=experiment,
                            output=output,
                            original_size=original_size,
                            split_layer_index=split_layer_index,
                        )

                        # Update metrics
                        self.metrics.update(processing_time)

                    logger.debug(f"Processed data in {processing_time:.4f}s")

                    # Compress the processed result to send back
                    compressed_result, result_size = self.compress_data.compress_data(
                        processed_result
                    )

                    # Send result back to client
                    self._send_result(
                        conn, result_size, processing_time, compressed_result
                    )
                    logger.debug(
                        f"Sent result of size {result_size} bytes back to client"
                    )

                except Exception as e:
                    logger.error(f"Error processing client data: {e}", exc_info=True)
                    break

    def _send_result(
        self,
        conn: socket.socket,
        result_size: int,
        processing_time: float,
        compressed_result: bytes,
    ) -> None:
        """
        Send the processed result back to the client using framed protocol.

        The response protocol uses:
        1. 4-byte length prefix for result size
        2. 4-byte field for processing time (as padded string)
        3. Variable-length compressed result data
        """
        try:
            # Send result size as header (4 bytes)
            size_bytes = result_size.to_bytes(LENGTH_PREFIX_SIZE, "big")
            conn.sendall(size_bytes)

            # Send processing time as fixed-length bytes (4 bytes)
            # Format as a string, pad/truncate to exactly 4 bytes
            time_str = str(processing_time).ljust(LENGTH_PREFIX_SIZE)
            time_bytes = time_str[:LENGTH_PREFIX_SIZE].encode()
            conn.sendall(time_bytes)

            # Send compressed result data
            conn.sendall(compressed_result)

        except Exception as e:
            logger.error(f"Error sending result: {e}")
            raise

    def _update_compression(self, config: dict) -> None:
        """
        Update compression settings from received configuration.

        Compression settings affect the tradeoff between:
        - Network bandwidth usage
        - CPU utilization for compression/decompression
        - Memory usage during transfer
        """
        # Check encryption mode from config (default to "transmission")
        encryption_mode = config.get("encryption", {}).get("mode", "transmission")
        
        if "compression" in config:
            logger.debug(f"Updating compression settings: {config['compression']}")
            
            # Create new compression instance with the received settings
            if self.encryption:
                # If we have encryption set up, pass it to the new compression instance
                self.compress_data = DataCompression(
                    config["compression"], 
                    encryption=self.encryption,
                    encryption_mode=encryption_mode
                )
                if encryption_mode == "full":
                    logger.info("Updated compression with full encryption mode - server will process encrypted tensors")
                else:
                    logger.debug("Updated compression with transmission encryption mode")
            else:
                # Standard compression without encryption
                self.compress_data = DataCompression(config["compression"])
                logger.debug("Updated compression without encryption")
        else:
            # If no compression settings in config, keep current settings
            if self.encryption:
                # But make sure we use the encryption if it's been set up
                self.compress_data = DataCompression(
                    SERVER_COMPRESSION_SETTINGS, 
                    encryption=self.encryption,
                    encryption_mode=encryption_mode
                )
                if encryption_mode == "full":
                    logger.info("Using minimal compression with full encryption mode")
                else:
                    logger.debug("Keeping minimal compression settings with transmission encryption")
            else:
                logger.warning("No compression settings in config, keeping minimal settings without encryption")

    def cleanup(self) -> None:
        """
        Clean up server resources and close the socket.

        Ensures graceful shutdown with proper resource release
        and final metrics logging.
        """
        logger.info("Starting server cleanup...")
        if self.server_socket:
            try:
                self.server_socket.shutdown(socket.SHUT_RDWR)
                self.server_socket.close()
                self.server_socket = None
                logger.info("Server socket cleaned up")
            except Exception as e:
                logger.error(f"Error during socket cleanup: {e}")

        if logging_server:
            shutdown_logging_server(logging_server)

        # Log final metrics if any requests were processed
        if self.metrics.total_requests > 0:
            logger.info(
                f"Final metrics: {self.metrics.total_requests} requests processed, "
                f"average processing time: {self.metrics.avg_processing_time:.4f}s"
            )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run server for split computing")
    parser.add_argument(
        "-l",
        "--local",
        action="store_true",
        help="Run experiment locally instead of as a network server",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to configuration file (required for local mode)",
        required=False,
    )
    
    # Add encryption options
    encryption_group = parser.add_argument_group('Encryption Options')
    encryption_group.add_argument(
        "--encrypt",
        action="store_true",
        help="Enable tensor encryption for secure transmission",
    )
    encryption_group.add_argument(
        "--encryption-key-file",
        type=str,
        help="Path to encryption key file (will override SPLIT_COMPUTE_ENCRYPTION_KEY env var)",
    )
    encryption_group.add_argument(
        "--encryption-password",
        type=str,
        help="Password for encryption (alternative to key file)",
    )
    encryption_group.add_argument(
        "--encryption-degree",
        type=int,
        default=8192,
        help="Polynomial modulus degree for encryption (default: 8192)",
    )
    encryption_group.add_argument(
        "--encryption-scale",
        type=int,
        default=26,
        help="Bit scale for encryption precision (default: 26)",
    )
    
    args = parser.parse_args()

    if args.local and not args.config:
        parser.error("--config is required when running in local mode")

    return args


if __name__ == "__main__":
    args = parse_arguments()

    server = Server(
        local_mode=args.local, 
        config_path=args.config,
        enable_encryption=args.encrypt,
        encryption_key_file=args.encryption_key_file,
        encryption_password=args.encryption_password,
        encryption_degree=args.encryption_degree,
        encryption_scale=args.encryption_scale
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down server due to keyboard interrupt...")
    except Exception as e:
        logger.error(f"Server crashed with error: {e}", exc_info=True)
    finally:
        server.cleanup()
