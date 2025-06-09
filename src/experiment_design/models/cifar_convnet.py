"""CIFAR-10 models compatible with tracr framework

HE-Friendly Model Adaptation:
============================

This model automatically adapts to homomorphic encryption (HE) mode when 
"full" encryption is enabled in the configuration. 

To enable HE-friendly mode, set in your config file (e.g., config/cifar10.yaml):

    encryption:
      enabled: true
      mode: "full"    # This triggers HE-friendly adaptations
      
The model will automatically:
- Use square activation (x²) instead of ReLU (HE-compatible)  
- Use AvgPool2d instead of MaxPool2d (linear operation)
- Disable dropout (not deterministic/HE-compatible)
- Keep the same architecture for seamless operation

Usage:
------
# Standard mode (ReLU, MaxPool, with dropout)
encryption:
  enabled: false
  
# HE-friendly mode (square activation, AvgPool, no dropout)  
encryption:
  enabled: true
  mode: "full"

The model detects the encryption mode automatically and adapts its operations
while maintaining the same network architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
import numpy as np
from typing import Union
import os

from .core.registry import ModelRegistry

logger = logging.getLogger("split_computing_logger")


@ModelRegistry.register("cifar_convnet")
class CIFARConvNet(nn.Module):
    """CIFAR-10 ConvNet with conditional HE-friendly operations.
    
    Supports both standard and homomorphic encryption (HE) friendly modes:
    - Standard mode: Uses ReLU activations, MaxPool2d, and full batch normalization
    - HE-friendly mode: Uses square activations, AvgPool2d, and simplified operations
    """
    
    def __init__(self, model_config=None, dataset_config=None, **kwargs):
        """Initialize the model.
        
        Args:
            model_config: Configuration dictionary containing model parameters
                - use_he_friendly: Boolean to enable HE-compatible operations
                - encryption: Dict containing encryption settings
            dataset_config: Dataset configuration (not used but may be passed)
            **kwargs: Additional keyword arguments, may include 'full_config'
        """
        super(CIFARConvNet, self).__init__()
        
        # Extract parameters from model_config
        if model_config is None:
            model_config = {}
        
        pretrained = model_config.get("pretrained", True)
        weight_path = model_config.get("weight_path")
        
        # Determine if HE-friendly mode should be used
        # Check multiple possible ways HE mode could be specified
        self.use_he_friendly = False
        
        # Method 1: Direct parameter
        if model_config.get("use_he_friendly", False):
            self.use_he_friendly = True
            
        # Method 2: Check encryption config for "full" mode (in model_config)
        encryption_config = model_config.get("encryption", {})
        if encryption_config.get("enabled", False) and encryption_config.get("mode") == "full":
            self.use_he_friendly = True
            logger.info("Enabling HE-friendly mode due to full encryption configuration")
            
        # Method 3: Check if global encryption mode is set to full
        if hasattr(model_config, 'get') and model_config.get("global_encryption_mode") == "full":
            self.use_he_friendly = True
            
        # Method 4: Check full config if passed through kwargs (NEW!)
        full_config = kwargs.get('full_config') or kwargs.get('config')
        if full_config and isinstance(full_config, dict):
            full_encryption_config = full_config.get("encryption", {})
            if full_encryption_config.get("enabled", False) and full_encryption_config.get("mode") == "full":
                self.use_he_friendly = True
                logger.info("Enabling HE-friendly mode due to full encryption configuration in kwargs")
                
        # Method 5: Environment variable fallback (NEW!)
        if os.environ.get("TENSOR_ENCRYPTION_MODE") == "full":
            self.use_he_friendly = True
            logger.info("Enabling HE-friendly mode due to environment variable TENSOR_ENCRYPTION_MODE=full")
            
        logger.info(f"CIFAR ConvNet initialized with HE-friendly mode: {self.use_he_friendly}")
        
        # First conv block
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second conv block
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third conv block  
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layers - conditional based on HE mode
        if self.use_he_friendly:
            # Average pooling is HE-friendly (linear operation)
            self.pool1 = nn.AvgPool2d(2, 2)
            self.pool2 = nn.AvgPool2d(2, 2) 
            self.pool3 = nn.AvgPool2d(2, 2)
            logger.info("Using AvgPool2d for HE compatibility")
        else:
            # Max pooling for better accuracy in standard mode
            self.pool1 = nn.MaxPool2d(2, 2)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.pool3 = nn.MaxPool2d(2, 2)
            logger.info("Using MaxPool2d for standard mode")
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        
        # Dropout - only used in standard mode
        if not self.use_he_friendly:
            self.dropout = nn.Dropout(0.25)
        else:
            self.dropout = None
            logger.info("Dropout disabled for HE compatibility")
            
        self.fc2 = nn.Linear(256, 10)
        
        # Load weights if not pretrained and weight_path is provided
        if not pretrained and weight_path:
            self._load_weights(weight_path)
        
    def _load_weights(self, weight_path):
        """Load weights from the specified path."""
        try:
            if Path(weight_path).exists():
                logger.info(f"Loading weights from {weight_path}")
                state_dict = torch.load(weight_path, map_location='cpu')
                
                # Handle different possible state dict formats
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                self.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded weights")
            else:
                logger.warning(f"Weight file not found: {weight_path}")
        except Exception as e:
            logger.warning(f"Failed to load weights from {weight_path}: {e}")
    
    def _he_friendly_activation(self, x):
        """HE-friendly square activation function.
        
        Square activation (x²) is polynomial and thus compatible with 
        homomorphic encryption schemes like CKKS.
        """
        return x * x
    
    def _standard_activation(self, x):
        """Standard ReLU activation for better accuracy."""
        return F.relu(x)
    
    def _apply_activation(self, x):
        """Apply appropriate activation based on mode."""
        if self.use_he_friendly:
            return self._he_friendly_activation(x)
        else:
            return self._standard_activation(x)
    
    def _apply_batch_norm_and_activation(self, x, bn_layer):
        """Apply batch normalization and activation.
        
        In HE mode, batch normalization is applied but may need special handling
        depending on the homomorphic encryption implementation.
        """
        x = bn_layer(x)
        return self._apply_activation(x)
        
    def forward(self, x):
        """Forward pass with conditional HE-friendly operations."""
        
        # First conv block
        x = self.conv1(x)
        x = self._apply_batch_norm_and_activation(x, self.bn1)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self._apply_batch_norm_and_activation(x, self.bn2)
        x = self.pool2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self._apply_batch_norm_and_activation(x, self.bn3)
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # First fully connected layer
        x = self.fc1(x)
        x = self._apply_batch_norm_and_activation(x, self.bn_fc1)
        
        # Apply dropout only in standard mode
        if not self.use_he_friendly and self.dropout is not None:
            x = self.dropout(x)
        
        # Final classification layer (no activation - raw logits)
        x = self.fc2(x)
        return x
    
    def get_he_mode(self):
        """Return whether the model is in HE-friendly mode."""
        return self.use_he_friendly
    
    def set_he_mode(self, he_friendly: bool):
        """Dynamically switch between HE-friendly and standard mode.
        
        Note: This only affects activation functions and dropout.
        Pooling layers are fixed at initialization time.
        """
        old_mode = self.use_he_friendly
        self.use_he_friendly = he_friendly
        
        if old_mode != he_friendly:
            logger.info(f"Switched CIFAR ConvNet mode: HE-friendly={he_friendly}")
            
        return old_mode 