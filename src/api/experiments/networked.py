"""
Networked distributed computing experiment using client-server tensor sharing.

This module implements the client side of a split computing architecture where
a neural network model is divided between local and remote processing.
"""

import logging
import psutil
import time
import os
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from tqdm import tqdm
import numpy as np
from PIL import Image

from .base import BaseExperiment, ProcessingTimes
from ..network import DataCompression, create_network_client
from ..network.encryption import TensorEncryption

logger = logging.getLogger("split_computing_logger")


class NetworkedExperiment(BaseExperiment):
    """Client-side distributed experiment that offloads computation to a server."""

    def __init__(
        self, 
        config: Dict[str, Any], 
        host: str, 
        port: int,
        encryption_password: Optional[str] = None,
        encryption_key_file: Optional[str] = None,
        encryption_degree: int = 8192,
        encryption_scale: int = 26
    ):
        """
        Initialize the networked experiment for distributed neural network processing.
        
        This class sets up:
        1. Model splitting configuration
        2. Network communication to server
        3. Tensor compression for efficient transmission
        4. Optional encryption for secure tensor sharing
        """
        super().__init__(config, host, port)

        logger.info(f"Initializing networked experiment with host={host}, port={port}")

        # Initialize layer timing data dictionary if not already initialized in parent class
        if not hasattr(self, "layer_timing_data"):
            self.layer_timing_data = {}

        # Set compression configuration for efficient tensor transmission
        self.compression_config = self.config.get("compression", {})
        if not self.compression_config:
            # Default compression configuration optimized for neural network tensors
            self.compression_config = {
                "clevel": 3,  # Compression level (higher = smaller size but slower)
                "filter": "SHUFFLE",  # Data pre-conditioning filter
                "codec": "ZSTD",  # Compression algorithm
            }

        # Store encryption parameters for lazy initialization (NO encryption during setup)
        self.encryption = None
        self.encryption_initialized = False
        
        # Store encryption settings without initializing anything heavy
        encryption_config = self.config.get("encryption", {})
        self.encryption_enabled = encryption_config.get("enabled", False)
        self.encryption_mode = encryption_config.get("mode", "transmission")
        self.encryption_password = encryption_config.get("password") or encryption_password
        self.encryption_key_file = encryption_config.get("key_file") or encryption_key_file
        self.encryption_degree = encryption_config.get("degree", encryption_degree)
        self.encryption_scale = encryption_config.get("scale", encryption_scale)
        
        # Check if encryption will be needed later (without initializing it now)
        self.encryption_requested = (self.encryption_enabled or 
                                   self.encryption_password is not None or 
                                   self.encryption_key_file is not None or
                                   encryption_degree != 8192 or encryption_scale != 26)
        
        # No logging about encryption during initialization to keep logs identical

        # Setup network client for tensor sharing with the server (no encryption during setup)
        try:
            logger.info(f"Creating network client to connect to {host}:{port}")
            # Create client without encryption during initialization
            self.network_client = create_network_client(
                config=self.config, 
                host=host, 
                port=port,
                encryption=None  # No encryption during setup
            )
            logger.info("Network client created successfully")
        except Exception as e:
            logger.error(f"Failed to create network client: {e}", exc_info=True)
            raise

        # Initialize data compression for efficient tensor transmission (no encryption during setup)
        try:
            compression_config = self.config.get("compression", {})
            logger.info(
                f"Initializing data compression with config: {compression_config}"
            )
            # Initialize without encryption during setup
            self.compress_data = DataCompression(
                compression_config,
                encryption=None,  # No encryption during setup
                encryption_mode="transmission"
            )
            logger.info("Data compression without encryption initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize data compression: {e}", exc_info=True)
            raise

        # Check if we can monitor battery usage for energy profiling
        self.can_monitor_battery = (
            hasattr(psutil, "sensors_battery") and psutil.sensors_battery() is not None
        )

        if self.can_monitor_battery and self.collect_metrics:
            self.initial_battery_percent = psutil.sensors_battery().percent
            logger.info(f"Initial battery percentage: {self.initial_battery_percent}%")

    def _ensure_encryption_initialized(self):
        """Lazily initialize encryption only when needed for tensor transmission."""
        if self.encryption_requested and not self.encryption_initialized:
            try:
                logger.info(f"Initializing tensor encryption for transmission in {self.encryption_mode} mode...")
                
                if self.encryption_password:
                    # Create encryption from password
                    self.encryption = TensorEncryption.from_password(
                        password=self.encryption_password,
                        mode=self.encryption_mode,
                        degree=self.encryption_degree,
                        scale=self.encryption_scale
                    )
                elif self.encryption_key_file:
                    logger.warning("Key file-based encryption not yet implemented with new TensorEncryption")
                    # For now, generate a new encryption instance
                    self.encryption = TensorEncryption(
                        mode=self.encryption_mode,
                        degree=self.encryption_degree,
                        scale=self.encryption_scale
                    )
                else:
                    # Generate new encryption instance (server-generated encryption)
                    logger.info("No password or key file provided, generating new encryption keys")
                    self.encryption = TensorEncryption(
                        mode=self.encryption_mode,
                        degree=self.encryption_degree,
                        scale=self.encryption_scale
                    )
                
                # Update network client and data compression with encryption
                self.network_client.encryption = self.encryption
                self.compress_data.encryption = self.encryption
                self.compress_data.encryption_mode = self.encryption_mode
                
                self.encryption_initialized = True
                logger.info(f"Tensor encryption initialized successfully with mode={self.encryption_mode}")
            except Exception as e:
                logger.error(f"Failed to initialize encryption: {e}", exc_info=True)
                # Continue without encryption
                logger.warning("Continuing without encryption due to initialization failure")
                self.encryption = None
                self.encryption_requested = False

    def process_single_image(
        self,
        inputs: torch.Tensor,
        class_idx: Any,
        image_file: str,
        split_layer: int,
        output_dir: Optional[Path],
    ) -> Optional[ProcessingTimes]:
        """Process a single image using distributed computation across the network.

        This method implements the core tensor sharing process between local device and server:
        1. Process tensor locally up to split_layer
        2. Compress and transmit intermediate tensor to server
        3. Server completes processing and returns results
        4. Process results locally and optionally save visualization
        """
        try:
            # ===== LAZY ENCRYPTION INITIALIZATION =====
            # Initialize encryption only when needed for tensor transmission
            self._ensure_encryption_initialized()
            
            # ===== ENCRYPTION PREPARATION =====
            original_inputs = inputs.clone() if self.encryption else None
            
            if self.encryption and self.encryption.mode == "full":
                logger.info(f"Using homomorphic encryption for end-to-end secure processing")
                self._input_encrypted = True
            elif self.encryption and self.encryption.mode == "transmission":
                logger.info(f"Using transmission encryption for secure data transfer")
                self._input_encrypted = True
            else:
                self._input_encrypted = False
            
            # ===== HOST DEVICE PROCESSING =====
            # Process the initial part of the model (up to split_layer) on the local device
            host_start = time.time()

            # Move input tensor to target device (CPU/GPU)
            inputs = inputs.to(self.device, non_blocking=True)

            # Generate intermediate tensor by running model up to split point
            output = self._get_model_output(inputs, split_layer)

            # Move inputs back to CPU for image reconstruction
            original_image = None
            if original_inputs is not None:
                # Use original unencrypted inputs for visualization
                original_image = self._get_original_image(original_inputs.cpu(), image_file)
            else:
                original_image = self._get_original_image(inputs.cpu(), image_file)

            # ===== TENSOR PREPARATION FOR TRANSMISSION =====
            # Package tensor with metadata needed by server for processing
            original_size = (
                self.post_processor.get_input_size(original_image)
                if original_image is not None
                else (0, 0)
            )
            
            # Create data package for transmission
            data_to_send = (output, original_size)

            # Compress (and encrypt if enabled) the tensor package
            compressed_output, output_size = self.compress_data.compress_data(
                data_to_send
            )
            logger.debug(f"Compressed tensor data size: {output_size} bytes")

            host_time = time.time() - host_start

            # ===== NETWORK TRANSMISSION =====
            # Transmit compressed tensor to server and receive processed results
            travel_start = time.time()
            try:
                # Ensure connection is established before sending data
                if not getattr(self.network_client, "connected", False):
                    success = self.network_client.connect()
                    if not success:
                        logger.error("Failed to connect to server")
                        return None

                # Send split layer index and tensor data to server, receive processed results
                processed_result, server_time = (
                    self.network_client.process_split_computation(
                        split_layer, compressed_output
                    )
                )
                
                # ===== DECRYPTION PHASE (if needed) =====
                if self._input_encrypted and self.encryption:
                    if self.encryption.mode == "full":
                        # For homomorphic encryption (full mode), the compression module
                        # returns the encrypted package structure. The result structure
                        # should be: (encrypted_tensor_data, original_size)
                        # where encrypted_tensor_data contains the encrypted result
                        logger.info("Processing homomorphic encryption result")
                        
                        # The compression module already handles the format conversion
                        # No additional decryption needed here - results remain encrypted
                        # for homomorphic processing downstream
                        logger.debug("Result processed for homomorphic computation")
                    else:
                        # For transmission mode, the compression module automatically
                        # decrypts the result - no additional processing needed
                        logger.debug("Result already decrypted by compression module")
                else:
                    logger.debug("No decryption needed - encryption not used")
            except Exception as e:
                logger.error(f"Network processing failed: {e}", exc_info=True)
                return None

            travel_end = time.time()

            # Calculate actual network transmission time by subtracting server processing time
            travel_time = (travel_end - travel_start) - server_time

            # ===== RESULT VISUALIZATION (OPTIONAL) =====
            if output_dir and self.config.get("default", {}).get("save_layer_images"):
                self._save_intermediate_results(
                    processed_result,
                    original_image,
                    class_idx,
                    image_file,
                    output_dir,
                )

            # Return comprehensive timing metrics for performance analysis
            return ProcessingTimes(
                host_time=host_time, travel_time=travel_time, server_time=server_time
            )

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return None

    def _get_model_output(self, inputs: torch.Tensor, split_layer: int) -> torch.Tensor:
        """Generate the intermediate tensor by running the model up to the split point.

        This is the first step in the tensor sharing process - generating the
        tensor that will be transmitted over the network to the server.
        """
        with torch.no_grad():
            # Execute only the local part of the model (up to split_layer)
            # The 'end' parameter signals the model to stop at the specified layer
            output = self.model(inputs, end=split_layer)

            # Some models return additional metadata along with the tensor
            if isinstance(output, tuple):
                output, _ = output

            return output

    def test_split_performance(
        self, split_layer: int, batch_size: int = 1, num_runs: int = 5
    ) -> Tuple[float, float, float, float, float]:
        """Test the tensor sharing pipeline performance at a specific split point.

        Evaluates the full distributed computation process including:
        - Local computation (up to split_layer)
        - Tensor transmission over network
        - Remote computation (after split_layer)
        - Optional energy consumption measurement
        """
        times = []
        # Create output directory for visualizations if configured
        split_dir = None
        if self.paths and self.paths.images_dir:
            split_dir = self.paths.images_dir / f"split_{split_layer}"
            split_dir.mkdir(exist_ok=True)
            logger.info(f"Saving split layer images to {split_dir}")
        else:
            logger.warning("No output directory configured. Images won't be saved.")

        # Initialize energy monitoring for this split layer if available
        if (
            self.collect_metrics
            and hasattr(self.model, "energy_monitor")
            and self.model.energy_monitor is not None
        ):
            try:
                # Start energy measurement for current split configuration
                if hasattr(self.model.energy_monitor, "start_split_measurement"):
                    self.model.energy_monitor.start_split_measurement(split_layer)
                else:
                    logger.debug(
                        f"Energy monitor doesn't support split measurements for layer {split_layer}"
                    )
            except Exception as e:
                logger.warning(f"Error starting split measurement: {e}")

        # Register current split point with metrics collector if available
        if (
            self.collect_metrics
            and hasattr(self.model, "metrics_collector")
            and self.model.metrics_collector
        ):
            try:
                self.model.metrics_collector.set_split_point(split_layer)
                logger.info(f"Set split point {split_layer} in metrics collector")
            except Exception as e:
                logger.warning(f"Error setting split point in metrics collector: {e}")

        if split_layer not in self.layer_timing_data:
            self.layer_timing_data[split_layer] = {}

        # Process dataset using distributed computation with tensors split at specified layer
        with torch.no_grad():
            for batch in tqdm(
                self.data_loader, desc=f"Processing at split {split_layer}"
            ):
                times.extend(self._process_batch(batch, split_layer, split_dir))

        # Calculate and report performance metrics
        if times:
            total_host = sum(t.host_time for t in times)
            total_travel = sum(t.travel_time for t in times)
            total_server = sum(t.server_time for t in times)

            # Collect energy consumption metrics if available
            total_battery_energy = 0.0
            if (
                self.collect_metrics
                and hasattr(self.model, "energy_monitor")
                and self.model.energy_monitor is not None
            ):
                if hasattr(self.model.energy_monitor, "get_battery_energy"):
                    battery_energy = self.model.energy_monitor.get_battery_energy()
                    # Ensure battery_energy has a valid value
                    total_battery_energy = (
                        0.0 if battery_energy is None else battery_energy
                    )
                    if total_battery_energy > 0:
                        logger.info(
                            f"Split layer {split_layer} used {total_battery_energy:.2f}mWh"
                        )

                        # Store energy data for performance analysis
                        if hasattr(self.model, "forward_info"):
                            if split_layer in self.model.forward_info:
                                self.model.forward_info[split_layer][
                                    "host_battery_energy_mwh"
                                ] = total_battery_energy
                else:
                    total_battery_energy = 0.0
                    logger.debug(
                        "Energy monitor doesn't support battery energy measurements"
                    )

            # Log performance summary including computation and network metrics
            self._log_performance_summary(total_host, total_travel, total_server)

            return split_layer, total_host, total_travel, total_server

        return split_layer, 0.0, 0.0, 0.0

    def _process_batch(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
        split_layer: int,
        split_dir: Path,
    ) -> List[ProcessingTimes]:
        """Process a batch of images through the distributed tensor sharing pipeline."""
        inputs, class_indices, image_files = batch
        return [
            result
            for result in (
                self.process_single_image(
                    input_tensor.unsqueeze(
                        0
                    ),  # Add batch dimension for single image processing
                    class_idx,
                    image_file,
                    split_layer,
                    split_dir if split_dir else None,
                )
                for input_tensor, class_idx, image_file in zip(
                    inputs, class_indices, image_files
                )
            )
            if result is not None
        ]

    def run_experiment(self) -> None:
        """Run complete experiment with tensor sharing and measure energy consumption."""
        try:
            # Execute the experiment with distributed tensor processing
            self.run()

            # Calculate total energy consumed during the experiment
            if self.collect_metrics and self.initial_battery_percent is not None:
                try:
                    battery = psutil.sensors_battery()
                    if battery and not battery.power_plugged:
                        percent_diff = self.initial_battery_percent - battery.percent
                        if percent_diff > 0:
                            # Convert battery percentage to energy consumption
                            TYPICAL_BATTERY_CAPACITY = 50000  # 50Wh in mWh units
                            host_battery_energy = (
                                percent_diff / 100.0
                            ) * TYPICAL_BATTERY_CAPACITY
                            logger.info(
                                f"Total experiment used {percent_diff:.2f}% battery ({host_battery_energy:.2f}mWh)"
                            )

                            # Store energy data for the current split configuration
                            if hasattr(self.model, "forward_info"):
                                split_layer = int(self.config["model"]["split_layer"])
                                self.model.forward_info[split_layer][
                                    "host_battery_energy_mwh"
                                ] = host_battery_energy
                except Exception as e:
                    logger.warning(
                        f"Error calculating battery energy in run_experiment: {e}"
                    )

        except Exception as e:
            logger.error(f"Error running experiment: {e}")
