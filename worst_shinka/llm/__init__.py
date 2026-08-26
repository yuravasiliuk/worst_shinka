"""OpenRouter connectivity and model validation"""

from .connect import OpenRouterError, select_models_for_mode, validate_openrouter_setup

__all__ = ["OpenRouterError", "select_models_for_mode", "validate_openrouter_setup"]