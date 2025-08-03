#!/usr/bin/env python3
"""
CIAF Development Environment Setup Script

This script sets up the development environment for CIAF.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def main():
    """Set up the CIAF development environment."""
    print("🚀 Setting up CIAF Development Environment")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version}")
    
    # Install development dependencies
    try:
        print("📦 Installing development dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-e", "."
        ])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"
        ])
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError:
        print("⚠️ Failed to install some dependencies")
    
    # Create necessary directories
    dirs_to_create = [
        "logs",
        "reports", 
        "exports",
        "temp_storage"
    ]
    
    for dir_name in dirs_to_create:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Created directory: {dir_name}")
    
    print("\n✅ Development environment setup complete!")
    print("\nNext steps:")
    print("1. Run tests: python -m pytest tests/")
    print("2. Check examples: cd examples && python basic/sklearn_example.py")
    print("3. View docs: cd docs && make html")


if __name__ == "__main__":
    main()
