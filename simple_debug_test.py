#!/usr/bin/env python3
"""Simple debug test for CIAF preprocessing integration."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from ciaf.wrappers.model_wrapper import CIAFModelWrapper

# Simple training data
training_data = [
    {"content": "This is good", "metadata": {"id": "1", "target": 1}},
    {"content": "This is bad", "metadata": {"id": "2", "target": 0}},
]

print("DEBUG: Creating LogisticRegression model...")
model = LogisticRegression(random_state=42)

print("DEBUG: Creating CIAF wrapper...")
wrapper = CIAFModelWrapper(
    model=model,
    model_name="DebugTest",
    enable_preprocessing=True
)

print("DEBUG: Starting training...")
try:
    snapshot = wrapper.train(
        dataset_id="debug_test",
        training_data=training_data,
        master_password="test123",  # Added missing parameter
        model_version="1.0.0"
    )
    print("SUCCESS: Training completed!")
    print(f"Snapshot: {snapshot.snapshot_id}")
    
    # Test inference
    print("\nDEBUG: Testing inference...")
    try:
        prediction, receipt = wrapper.predict("This is great!", model_version="1.0.0")
        print(f"SUCCESS: Inference completed!")
        print(f"Prediction: {prediction}")
        print(f"Receipt: {receipt.receipt_hash}")
        
        # Show enhanced info if available
        if hasattr(receipt, 'enhanced_info') and receipt.enhanced_info:
            print(f"Enhanced Features:")
            for feature_name, feature_data in receipt.enhanced_info.items():
                print(f"  {feature_name}: {feature_data}")
        else:
            print("No enhanced info available")
            
    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
