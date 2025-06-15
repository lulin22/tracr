"""ConvNet implementation adapted to follow registry interface"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from torch import Tensor

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
        
        # Check if HE-friendly mode should be used
        self.use_he_friendly = False
        
        # Method 1: Direct parameter
        if model_config.get("use_he_friendly", False):
            self.use_he_friendly = True
            
        # Method 2: Check encryption config for "full" mode
        encryption_config = model_config.get("encryption", {})
        if encryption_config.get("enabled", False) and encryption_config.get("mode") == "full":
            self.use_he_friendly = True
            logger.info("Enabling HE-friendly mode due to full encryption configuration")
            
        # Method 3: Check if global encryption mode is set to full
        if hasattr(model_config, 'get') and model_config.get("global_encryption_mode") == "full":
            self.use_he_friendly = True
            
        # Method 4: Check full config if passed through kwargs
        full_config = kwargs.get('full_config') or kwargs.get('config')
        if full_config and isinstance(full_config, dict):
            full_encryption_config = full_config.get("encryption", {})
            if full_encryption_config.get("enabled", False) and full_encryption_config.get("mode") == "full":
                self.use_he_friendly = True
                logger.info("Enabling HE-friendly mode due to full encryption configuration in kwargs")
        
        logger.info(f"ConvNet initialized with HE-friendly mode: {self.use_he_friendly}")
        
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
        """Return whether the model is compatible with homomorphic encryption."""
        return self.use_he_friendly
    
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Forward pass through the ConvNet model"""
        return self.model(x) 