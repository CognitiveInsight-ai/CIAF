#!/usr/bin/env python3
"""
CIAF Clean Build Script

Builds a clean distribution of CIAF ready for packaging and testing.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and check for errors."""
    print(f"📋 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   ✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Command failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False


def clean_build_artifacts():
    """Remove build artifacts."""
    print("🧹 Cleaning build artifacts...")
    
    artifacts = [
        "build/",
        "dist/",
        "*.egg-info/",
        ".pytest_cache/",
        "htmlcov/",
        "__pycache__/",
        "*.pyc",
        ".coverage",
        ".mypy_cache/",
        "temp_cleanup/"
    ]
    
    for pattern in artifacts:
        if "*" in pattern:
            # Use shell command for glob patterns
            if os.name == 'nt':  # Windows
                run_command(f'for /r . %i in ({pattern}) do @if exist "%i" rmdir /s /q "%i" 2>nul || del /q "%i" 2>nul', f"Removing {pattern}")
            else:  # Unix-like
                run_command(f'find . -name "{pattern}" -type f -delete', f"Removing {pattern}")
                run_command(f'find . -name "{pattern}" -type d -exec rm -rf {{}} + 2>/dev/null || true', f"Removing {pattern} directories")
        else:
            path = Path(pattern)
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                    print(f"   🗑️ Removed directory: {pattern}")
                else:
                    path.unlink()
                    print(f"   🗑️ Removed file: {pattern}")


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    
    # Install test dependencies if needed
    run_command(f"{sys.executable} -m pip install pytest pytest-cov", "Installing test dependencies")
    
    # Run tests
    success = run_command(f"{sys.executable} -m pytest tests/ -v --cov=ciaf --cov-report=term-missing", "Running test suite")
    
    if not success:
        print("❌ Tests failed. Please fix issues before building.")
        return False
    
    return True


def run_linting():
    """Run code quality checks."""
    print("🔍 Running code quality checks...")
    
    # Install linting tools if needed
    linting_tools = ["black", "isort", "flake8"]
    for tool in linting_tools:
        run_command(f"{sys.executable} -m pip install {tool}", f"Installing {tool}")
    
    # Format code
    run_command(f"{sys.executable} -m black ciaf/ tests/ examples/ --exclude temp_cleanup", "Formatting code with black")
    run_command(f"{sys.executable} -m isort ciaf/ tests/ examples/ --skip temp_cleanup", "Sorting imports with isort")
    
    # Check code quality
    success = run_command(f"{sys.executable} -m flake8 ciaf/ tests/ examples/ --exclude=temp_cleanup --max-line-length=88 --extend-ignore=E203,W503", "Running flake8 checks")
    
    if not success:
        print("⚠️ Code quality issues found. Consider fixing them.")
    
    return True


def build_package():
    """Build the package."""
    print("📦 Building package...")
    
    # Install build tools
    run_command(f"{sys.executable} -m pip install build twine", "Installing build tools")
    
    # Build package
    success = run_command(f"{sys.executable} -m build", "Building distribution packages")
    
    if not success:
        print("❌ Package build failed.")
        return False
    
    # Check package
    success = run_command(f"{sys.executable} -m twine check dist/*", "Checking package")
    
    return success


def create_real_world_test():
    """Create a real-world test script."""
    print("🌍 Creating real-world test script...")
    
    test_script = '''#!/usr/bin/env python3
"""
CIAF Real-World Test Script

Tests CIAF with a realistic machine learning pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

# Import CIAF components
from ciaf import ModelMetadataManager, DatasetAnchor, capture_metadata


def main():
    """Run a real-world test of CIAF."""
    print("🚀 CIAF Real-World Test")
    print("=" * 50)
    
    # Initialize CIAF metadata manager
    manager = ModelMetadataManager("real_world_test_model", "1.0.0")
    
    # Generate synthetic dataset
    print("📊 Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Create dataset anchor for provenance
    dataset_anchor = DatasetAnchor(X.tolist(), "synthetic_classification_dataset")
    print(f"   ✅ Dataset anchored with hash: {dataset_anchor.anchor_hash[:16]}...")
    
    # Log data ingestion
    manager.log_data_ingestion({
        "dataset_name": "synthetic_classification_dataset",
        "samples": len(X),
        "features": X.shape[1],
        "classes": len(np.unique(y)),
        "anchor_hash": dataset_anchor.anchor_hash
    })
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model with metadata tracking
    print("🎯 Training model...")
    
    training_config = {
        "algorithm": "RandomForestClassifier",
        "parameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        },
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    manager.log_training_start(training_config)
    
    # Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    training_results = {
        "accuracy": accuracy,
        "feature_importance": model.feature_importances_.tolist()[:5],  # Top 5
        "model_parameters": model.get_params()
    }
    
    manager.log_training_complete(training_results)
    
    print(f"   ✅ Model trained with accuracy: {accuracy:.4f}")
    
    # Test inference tracking
    print("🔮 Testing inference tracking...")
    
    # Make predictions with metadata logging
    sample_inference = X_test[:5]  # First 5 test samples
    predictions = model.predict(sample_inference)
    
    inference_info = {
        "samples_predicted": len(sample_inference),
        "predictions": predictions.tolist(),
        "model_version": "1.0.0"
    }
    
    manager.log_inference(inference_info)
    
    # Get complete pipeline trace
    print("📋 Retrieving pipeline trace...")
    trace = manager.get_pipeline_trace()
    
    print(f"   ✅ Pipeline trace contains {len(trace)} events")
    
    # Verify dataset integrity
    print("🔒 Verifying dataset integrity...")
    is_valid = dataset_anchor.verify_integrity(X.tolist())
    print(f"   ✅ Dataset integrity: {'VALID' if is_valid else 'INVALID'}")
    
    # Print summary
    print("\\n" + "=" * 50)
    print("📊 CIAF Real-World Test Summary")
    print("=" * 50)
    print(f"Model: {manager.model_name} v{manager.version}")
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Pipeline Events: {len(trace)}")
    print(f"Dataset Integrity: {'✅ VALID' if is_valid else '❌ INVALID'}")
    
    # Event breakdown
    event_types = {}
    for event in trace:
        event_type = event.get("event_type", "unknown")
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    print("\\nEvent Breakdown:")
    for event_type, count in event_types.items():
        print(f"  {event_type}: {count}")
    
    print("\\n🎉 CIAF real-world test completed successfully!")
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\\n✅ All tests passed!")
        else:
            print("\\n❌ Some tests failed!")
            exit(1)
    except Exception as e:
        print(f"\\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
'''
    
    with open("test_real_world.py", "w") as f:
        f.write(test_script)
    
    print("   ✅ Created test_real_world.py")


def main():
    """Main cleanup and build process."""
    print("🚀 CIAF Clean Build Process")
    print("=" * 50)
    
    # Step 1: Clean artifacts
    clean_build_artifacts()
    
    # Step 2: Run code quality checks
    run_linting()
    
    # Step 3: Run tests
    if not run_tests():
        print("❌ Build aborted due to test failures.")
        return False
    
    # Step 4: Build package
    if not build_package():
        print("❌ Build aborted due to packaging errors.")
        return False
    
    # Step 5: Create real-world test
    create_real_world_test()
    
    print("\n" + "=" * 50)
    print("✅ CIAF Clean Build Completed Successfully!")
    print("=" * 50)
    print("\n📦 Package artifacts:")
    print("   📁 dist/ - Contains built packages")
    print("   🐍 test_real_world.py - Real-world test script")
    
    print("\n🚀 Next Steps:")
    print("1. Install the package: pip install dist/ciaf-*.whl")
    print("2. Run real-world test: python test_real_world.py")
    print("3. Test with your own models using CIAF integration")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
