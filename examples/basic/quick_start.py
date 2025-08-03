#!/usr/bin/env python3
"""
CIAF Quick Start Example

This example demonstrates the basic usage of CIAF with a simple scikit-learn model.
Shows the minimal setup needed to add CIAF tracking to existing ML workflows.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import CIAF wrapper
from ciaf.wrappers import CIAFModelWrapper


def main():
    """Demonstrate basic CIAF usage."""
    print("🚀 CIAF Quick Start Example")
    print("=" * 40)
    
    # Generate sample data
    print("📊 Generating sample data...")
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✅ Data prepared: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    
    # Create standard scikit-learn model
    print("\n🔧 Creating model...")
    base_model = LogisticRegression(random_state=42)
    
    # Wrap with CIAF for full transparency and provenance tracking
    print("🛡️ Wrapping with CIAF...")
    model = CIAFModelWrapper(
        model=base_model,
        model_id="quick_start_classifier"
    )
    
    # Train model (CIAF automatically captures training metadata)
    print("🎯 Training model...")
    model.fit(X_train, y_train)
    print("✅ Training complete with CIAF provenance tracking")
    
    # Make predictions (CIAF automatically captures inference metadata)
    print("\n🔮 Making predictions...")
    predictions = model.predict(X_test)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, predictions)
    print(f"📈 Model accuracy: {accuracy:.3f}")
    
    # Generate CIAF compliance report
    print("\n📋 CIAF automatically tracked:")
    print("  • Training data fingerprint")
    print("  • Model parameters and hyperparameters") 
    print("  • Training metadata and timestamps")
    print("  • Inference requests and responses")
    print("  • Cryptographic integrity verification")
    
    # Get model metadata summary
    try:
        metadata = model.get_training_metadata()
        print(f"\n🔍 Training metadata captured:")
        print(f"  • Training samples: {metadata.get('training_samples', 'N/A')}")
        print(f"  • Features: {metadata.get('feature_count', 'N/A')}")
        print(f"  • Model type: {metadata.get('model_type', 'N/A')}")
        print(f"  • Training time: {metadata.get('training_duration', 'N/A')}")
    except Exception as e:
        print(f"⚠️ Could not retrieve metadata: {e}")
    
    print("\n✅ CIAF Quick Start Complete!")
    print("\nNext steps:")
    print("  • Try examples/compliance/ for regulatory compliance demos")
    print("  • See examples/industry/ for domain-specific examples")
    print("  • Check examples/advanced/ for customization options")


if __name__ == "__main__":
    main()
