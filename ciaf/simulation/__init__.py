"""
CIAF Simulation Package

Provides mock implementations for testing and demonstration of the
Cognitive Insight AI Framework (CIAF) components.
"""

from .ml_framework import MLFrameworkSimulator
from .mock_llm import MockLLM

__all__ = ["MockLLM", "MLFrameworkSimulator"]
