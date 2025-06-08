"""Implementation of a model that can be hooked into a split computing framework"""

from .model_hooked import WrappedModel

# Import model implementations to ensure they are registered
from . import custom  # Registers "custom" model
from . import mnist_convnet  # Registers "mnist_convnet" model
from . import cifar_convnet  # Registers "cifar_convnet" model

__all__ = ["WrappedModel"]
