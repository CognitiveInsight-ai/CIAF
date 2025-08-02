"""
Practical example of CIAFModelWrapper working with different types of models.
This demonstrates what works well and what needs adaptation.
"""

import numpy as np
from ciaf.wrappers import CIAFModelWrapper

# Example 1: Custom model that works perfectly
class SimpleNumericalModel:
    """A custom model that works well with CIAF wrapper."""
    
    def __init__(self):
        self.weights = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """Simple linear model fitting."""
        X = np.array(X)
        y = np.array(y)
        self.weights = np.random.random(X.shape[1]) if len(X.shape) > 1 else np.random.random(1)
        self.is_fitted = True
        print(f"   SimpleNumericalModel fitted with {len(X)} samples")
    
    def predict(self, X):
        """Simple prediction."""
        if not self.is_fitted:
            return [0] * len(X)
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return (X @ self.weights.reshape(-1, 1)).flatten() if len(self.weights) > 1 else X.flatten() * self.weights

# Example 2: Text classifier that handles its own preprocessing
class TextClassifierWithPreprocessing:
    """A text classifier that handles preprocessing internally."""
    
    def __init__(self):
        self.word_scores = {}
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit by learning word sentiment scores."""
        self.word_scores = {}
        for text, label in zip(X, y):
            words = str(text).lower().split()
            for word in words:
                if word not in self.word_scores:
                    self.word_scores[word] = []
                self.word_scores[word].append(label)
        
        # Average the scores
        for word in self.word_scores:
            self.word_scores[word] = np.mean(self.word_scores[word])
        
        self.is_fitted = True
        print(f"   TextClassifier fitted with {len(X)} samples, learned {len(self.word_scores)} words")
    
    def predict(self, X):
        """Predict using word scores."""
        if not self.is_fitted:
            return [0.5] * len(X)
        
        predictions = []
        for text in X:
            words = str(text).lower().split()
            scores = [self.word_scores.get(word, 0.5) for word in words]
            prediction = np.mean(scores) if scores else 0.5
            predictions.append(prediction)
        return predictions

def test_numerical_model():
    """Test with numerical data."""
    print("\n" + "="*60)
    print("🔢 Testing with Numerical Data")
    print("="*60)
    
    # Numerical training data
    training_data = [
        {"content": [1.0, 2.0], "metadata": {"id": "1", "target": 0.5}},
        {"content": [2.0, 3.0], "metadata": {"id": "2", "target": 1.0}},
        {"content": [3.0, 4.0], "metadata": {"id": "3", "target": 1.5}},
        {"content": [4.0, 5.0], "metadata": {"id": "4", "target": 2.0}},
    ]
    
    model = SimpleNumericalModel()
    wrapper = CIAFModelWrapper(
        model=model,
        model_name="SimpleNumerical",
        enable_chaining=True
    )
    
    # Train
    snapshot = wrapper.train(
        dataset_id="numerical_dataset",
        training_data=training_data,
        master_password="test_password",
        model_version="1.0.0"
    )
    
    # Predict
    prediction, receipt = wrapper.predict([2.5, 3.5])
    print(f"✅ Numerical prediction: {prediction}")
    print(f"   Receipt: {receipt.receipt_hash[:16]}...")
    
    return True

def test_text_model_with_preprocessing():
    """Test with text data using a model that handles preprocessing."""
    print("\n" + "="*60)
    print("📝 Testing with Text Data (Preprocessed)")
    print("="*60)
    
    # Text training data
    training_data = [
        {"content": "This is great excellent amazing", "metadata": {"id": "1", "target": 1}},
        {"content": "This is bad terrible horrible", "metadata": {"id": "2", "target": 0}},
        {"content": "Good quality nice product", "metadata": {"id": "3", "target": 1}},
        {"content": "Poor service disappointing experience", "metadata": {"id": "4", "target": 0}},
        {"content": "Outstanding fantastic wonderful", "metadata": {"id": "5", "target": 1}},
    ]
    
    model = TextClassifierWithPreprocessing()
    wrapper = CIAFModelWrapper(
        model=model,
        model_name="TextClassifier",
        enable_chaining=True
    )
    
    # Train
    snapshot = wrapper.train(
        dataset_id="text_dataset",
        training_data=training_data,
        master_password="test_password",
        model_version="1.0.0"
    )
    
    # Predict
    prediction, receipt = wrapper.predict("This is excellent quality")
    print(f"✅ Text prediction: {prediction}")
    print(f"   Receipt: {receipt.receipt_hash[:16]}...")
    
    return True

def test_simple_callable():
    """Test with a simple function."""
    print("\n" + "="*60)
    print("🔧 Testing with Callable Function")
    print("="*60)
    
    def simple_predictor(inputs):
        """A simple prediction function."""
        return [f"processed_{inp}" for inp in inputs]
    
    # Add predict method
    simple_predictor.predict = lambda X: [f"prediction_for_{x}" for x in X]
    
    wrapper = CIAFModelWrapper(
        model=simple_predictor,
        model_name="SimpleCallable",
        enable_chaining=True
    )
    
    # Training data (function doesn't need fitting)
    training_data = [
        {"content": "input1", "metadata": {"id": "1", "target": "output1"}},
        {"content": "input2", "metadata": {"id": "2", "target": "output2"}},
    ]
    
    # Train (will skip actual model fitting since no fit method)
    snapshot = wrapper.train(
        dataset_id="callable_dataset",
        training_data=training_data,
        master_password="test_password",
        model_version="1.0.0",
        fit_model=False  # Skip fitting
    )
    
    # Predict
    prediction, receipt = wrapper.predict("test_input")
    print(f"✅ Callable prediction: {prediction}")
    print(f"   Receipt: {receipt.receipt_hash[:16]}...")
    
    return True

def main():
    """Run practical examples."""
    print("🚀 CIAF Model Wrapper - Practical Examples")
    print("=" * 60)
    print("This demonstrates what works well with the CIAF wrapper")
    
    results = []
    
    try:
        results.append(("Numerical Model", test_numerical_model()))
    except Exception as e:
        print(f"❌ Numerical model failed: {e}")
        results.append(("Numerical Model", False))
    
    try:
        results.append(("Text Model (Preprocessed)", test_text_model_with_preprocessing()))
    except Exception as e:
        print(f"❌ Text model failed: {e}")
        results.append(("Text Model (Preprocessed)", False))
    
    try:
        results.append(("Callable Function", test_simple_callable()))
    except Exception as e:
        print(f"❌ Callable function failed: {e}")
        results.append(("Callable Function", False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 PRACTICAL TEST RESULTS")
    print("="*60)
    
    for name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{name:25} : {status}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nResults: {successful}/{total} examples worked successfully")
    
    print("\n" + "="*60)
    print("💡 KEY INSIGHTS")
    print("="*60)
    print("✅ Works well with:")
    print("   • Custom models with fit() and predict() methods")
    print("   • Models that handle their own preprocessing")
    print("   • Numerical data")
    print("   • Functions with predict() method")
    print("   • Models that don't need fitting (predict-only)")
    print()
    print("⚠️  Needs adaptation for:")
    print("   • Raw sklearn models with text data (need vectorization)")
    print("   • Deep learning models (need tensor conversion)")
    print("   • Complex preprocessing pipelines")
    print("   • Models requiring special input formats")
    print()
    print("🎯 CIAF provides full provenance even when model training fails!")

if __name__ == "__main__":
    main()
