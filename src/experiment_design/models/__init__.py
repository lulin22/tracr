"""Implementation of a model that can be hooked into a split computing framework"""

from .model_hooked import WrappedModel

# Import model implementations to ensure they are registered
from . import custom  # Registers "custom" model
from . import network1_adapted  # Registers "convnet" model

__all__ = ["WrappedModel"]
