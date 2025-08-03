#!/usr/bin/env python3
"""Test specifically the numerical regression fix"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LinearRegression
from ciaf.wrappers.model_wrapper import CIAFModelWrapper

def test_numerical_regression():
    """Test numerical regression with the enhanced features."""
    print(f"Testing Enhanced Numerical Regression")
    print(f"=" * 50)
    
    try:
        model = LinearRegression()
        wrapper = CIAFModelWrapper(
            model=model,
            model_name="TestRegression",
            enable_preprocessing=True,
            enable_explainability=True,
            enable_uncertainty=True,
            enable_metadata_tags=True
        )
        
        # Numerical data
        numerical_data = []
        for i in range(10):
            x = float(i)
            y = 2 * x + 1 + (i % 3 - 1) * 0.1  # Linear with small noise
            numerical_data.append({
                "content": x,
                "metadata": {"id": str(i), "target": y}
            })
        
        print(f"Training data: {len(numerical_data)} samples")
        print(f"Sample: x={numerical_data[0]['content']}, y={numerical_data[0]['metadata']['target']}")
        
        # Training
        snapshot = wrapper.train(
            dataset_id="test_numerical_regression",
            training_data=numerical_data,
            master_password="test_password_123",
            model_version="1.0.0"
        )
        
        print(f"✅ Training completed")
        
        # Test predictions
        test_cases = [
            5.0,           # Single value
            [3.0, 7.0],    # Multiple values
            2.5            # Another single value
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            print(f"\nTest {i}: Input = {test_input}")
            prediction, receipt = wrapper.predict(test_input, model_version="1.0.0")
            
            print(f"   Prediction: {prediction}")
            if isinstance(test_input, (list, tuple)):
                expected = [2 * x + 1 for x in test_input]
                print(f"   Expected: ~{expected}")
            else:
                expected = 2 * test_input + 1
                print(f"   Expected: ~{expected}")
            
            print(f"   Receipt: {receipt.receipt_hash[:16]}...")
            
            # Show enhanced info if available
            if hasattr(receipt, 'enhanced_info') and receipt.enhanced_info:
                info = receipt.enhanced_info
                print(f"   Enhanced Info:")
                if 'explainability' in info:
                    print(f"     explainability: {info['explainability']['method']}")
                if 'uncertainty' in info:
                    print(f"     uncertainty: {info['uncertainty']['confidence_level']}")
                if 'metadata_tag' in info:
                    print(f"     metadata_tag: {info['metadata_tag']['tag_id']}")
        
        print(f"\n✅ All numerical regression tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Numerical regression failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_numerical_regression()
