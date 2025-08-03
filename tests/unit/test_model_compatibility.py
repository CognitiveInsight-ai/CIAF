"""
Test script to verify CIAFModelWrapper compatibility with different ML models.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from ciaf.wrappers import CIAFModelWrapper

# Sample training data in CIAF format
training_data = [
    {"content": "This is a positive review", "metadata": {"id": "1", "target": 1}},
    {"content": "This is a negative review", "metadata": {"id": "2", "target": 0}},
    {"content": "Great product quality", "metadata": {"id": "3", "target": 1}},
    {"content": "Poor customer service", "metadata": {"id": "4", "target": 0}},
    {"content": "Excellent value for money", "metadata": {"id": "5", "target": 1}},
]

def test_model_compatibility(model, model_name):
    """Test a specific model with the CIAF wrapper."""
    print(f"\n{'='*50}")
    print(f"Testing {model_name}")
    print(f"{'='*50}")
    
    try:
        # Create wrapper
        wrapper = CIAFModelWrapper(
            model=model,
            model_name=f"Test{model_name}",
            enable_chaining=True
        )
        
        # Test training
        snapshot = wrapper.train(
            dataset_id=f"test_dataset_{model_name.lower()}",
            training_data=training_data,
            master_password="test_password_123",
            model_version="1.0.0"
        )
        
        print(f"✅ Training successful for {model_name}")
        
        # Test prediction
        prediction, receipt = wrapper.predict("This is a test review")
        print(f"✅ Prediction successful for {model_name}: {prediction}")
        print(f"   Receipt ID: {receipt.receipt_hash[:16]}...")
        
        # Test verification
        verification = wrapper.verify(receipt)
        print(f"✅ Verification successful for {model_name}: {verification['receipt_integrity']}")
        
        # Test model info
        info = wrapper.get_model_info()
        print(f"   Model info: {info['model_type']}, trained: {info['is_trained']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with {model_name}: {str(e)}")
        return False

def test_custom_model():
    """Test with a custom model class."""
    
    class CustomMLModel:
        def __init__(self):
            self.is_fitted = False
            
        def fit(self, X, y):
            self.is_fitted = True
            print(f"   Custom model fitted with {len(X)} samples")
            
        def predict(self, X):
            if not self.is_fitted:
                raise RuntimeError("Model not fitted")
            return ["positive" if i % 2 == 0 else "negative" for i in range(len(X))]
    
    return test_model_compatibility(CustomMLModel(), "CustomModel")

def test_simple_callable():
    """Test with a simple callable (function)."""
    
    def simple_predictor(inputs):
        return ["prediction"] * len(inputs)
    
    # Add predict method to make it compatible
    simple_predictor.predict = lambda x: ["simple_prediction"] * len(x)
    
    return test_model_compatibility(simple_predictor, "CallableModel")

def main():
    """Run all compatibility tests."""
    print("🧪 CIAF Model Wrapper Compatibility Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test scikit-learn models
    models_to_test = [
        (LogisticRegression(max_iter=1000), "LogisticRegression"),
        (RandomForestClassifier(n_estimators=10, random_state=42), "RandomForest"),
        (GaussianNB(), "NaiveBayes"),
    ]
    
    for model, name in models_to_test:
        results[name] = test_model_compatibility(model, name)
    
    # Test custom models
    results["CustomModel"] = test_custom_model()
    results["CallableModel"] = test_simple_callable()
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 COMPATIBILITY TEST RESULTS")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for model_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{model_name:20} : {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} models compatible")
    
    if passed_tests == total_tests:
        print("🎉 All models are compatible with CIAFModelWrapper!")
    else:
        print("⚠️  Some models may need additional adaptation.")

if __name__ == "__main__":
    main()
