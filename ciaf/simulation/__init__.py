"""
CIAF Simulation Package

Provides mock implementations for testing and demonstration of the
Cognitive Insight AI Framework (CIAF) components.
"""

from .mock_llm import MockLLM
from .ml_framework import MLFrameworkSimulator

__all__ = ['MockLLM', 'MLFrameworkSimulator']
