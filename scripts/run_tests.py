#!/usr/bin/env python3
"""
CIAF Test Runner

Unified test runner for CIAF with different test categories.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_tests(test_type="all", verbose=False):
    """Run CIAF tests."""
    
    # Base pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    # Add coverage
    cmd.extend(["--cov=ciaf", "--cov-report=html", "--cov-report=term"])
    
    # Determine test paths
    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    elif test_type == "all":
        cmd.append("tests/")
    else:
        cmd.append(f"tests/{test_type}/")
    
    print(f"🧪 Running {test_type} tests...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ All tests passed!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed with exit code {e.returncode}")
        return e.returncode


def main():
    parser = argparse.ArgumentParser(description="Run CIAF tests")
    parser.add_argument(
        "test_type", 
        nargs="?", 
        default="all",
        choices=["all", "unit", "integration", "performance"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    return run_tests(args.test_type, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
