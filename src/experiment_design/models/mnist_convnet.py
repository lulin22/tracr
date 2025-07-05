"""ConvNet implementation adapted to follow registry interface"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from torch import Tensor
import os

# Import the registry for model registration
from src.experiment_design.models.core.registry import ModelRegistry

logger = logging.getLogger("split_computing_logger")


class ConvNet(nn.Module):
    """Original ConvNet architecture with square activation functions"""
    
    def __init__(self, hidden=64, output=10):
        super(ConvNet, self).__init__()        
        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = nn.Linear(256, hidden)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv1(x)
        # the model uses the square activation function
        x = x * x
        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x


@ModelRegistry.register("mnist_convnet")
class ConvNetModel(nn.Module):
    """ConvNet model adapted to follow registry interface"""
    
    def __init__(self, model_config: Dict[str, Any], **kwargs: Any) -> None:
        """Initialize ConvNet model with registry configuration system"""
        super().__init__()
        
        # Extract ConvNet-specific parameters from model config
        self.hidden_size = model_config.get("hidden_size", 64)
        self.num_classes = model_config.get("num_classes", 10)
        self.input_size = tuple(model_config.get("input_size", [1, 28, 28]))
        
        # Default to non-HE mode
        self.use_he_friendly = False
        
        # Check for HE configuration in multiple ways
        if model_config and isinstance(model_config, dict):
            # Method 1: Direct model config
            if model_config.get("use_he_friendly", False):
                self.use_he_friendly = True
                logger.info("Enabling HE-friendly mode from model config")
            
            # Method 2: Global encryption mode
            if model_config.get("global_encryption_mode") == "full":
                self.use_he_friendly = True
                logger.info("Enabling HE-friendly mode from global encryption mode")
        
        # Method 3: Full config in kwargs
        full_config = kwargs.get('full_config') or kwargs.get('config')
        if full_config and isinstance(full_config, dict):
            encryption_config = full_config.get("encryption", {})
            if encryption_config.get("enabled", False) and encryption_config.get("mode") == "full":
                self.use_he_friendly = True
                logger.info("Enabling HE-friendly mode from full config")
        
        # Method 4: Environment variable
        if os.environ.get("TENSOR_ENCRYPTION_MODE") == "full":
            self.use_he_friendly = True
            logger.info("Enabling HE-friendly mode from environment variable")
        
        # Set activation based on HE mode
        self.activation = 'square' if self.use_he_friendly else 'relu'
        logger.info(f"Using {self.activation} activation for {'HE-friendly' if self.use_he_friendly else 'regular'} mode")
        
        # Create the actual ConvNet model
        self.model = ConvNet(hidden=self.hidden_size, output=self.num_classes)
        
        # Load weights if weight_path is specified
        if model_config.get("weight_path"):
            weight_path = model_config["weight_path"]
            try:
                from pathlib import Path
                if Path(weight_path).exists():
                    logger.info(f"Loading ConvNet weights from: {weight_path}")
                    device = model_config.get("default", {}).get("device", "cpu")
                    state_dict = torch.load(weight_path, map_location=device)
                    
                    # Load weights into the nested model
                    self.model.load_state_dict(state_dict)
                    logger.info("✓ ConvNet weights loaded successfully")
                else:
                    logger.warning(f"Weight file not found: {weight_path}")
            except Exception as e:
                logger.error(f"Failed to load ConvNet weights: {e}")
                logger.warning("Continuing with random weights")
        else:
            logger.info("No weight_path specified, using random weights")
        
        logger.info(f"ConvNet initialized with hidden_size={self.hidden_size}, num_classes={self.num_classes}")
    
    @property
    def is_he_compatible(self) -> bool:
        """Check if the model is configured for homomorphic encryption compatibility."""
        # Check if we're in HE-friendly mode
        if not hasattr(self, 'use_he_friendly'):
            return False
            
        # Check if we're using square activation (required for HE)
        if not hasattr(self, 'activation') or self.activation != 'square':
            return False
            
        return self.use_he_friendly
    
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Forward pass through the ConvNet model"""
        return self.model(x) 