"""
Complete example showing CIAF wrapper with scikit-learn models.
This demonstrates the proper way to use sklearn models with CIAF.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ciaf.wrappers import CIAFModelWrapper


def test_sklearn_with_preprocessing():
    """Test sklearn model with proper text preprocessing."""
    print("🔧 Testing Sklearn Model with Text Preprocessing")
    print("=" * 60)

    # Create a pipeline that handles text preprocessing
    sklearn_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=100)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    # Training data for sklearn (preprocessed)
    training_texts = [
        "This is great excellent amazing",
        "This is bad terrible horrible",
        "Good quality nice product",
        "Poor service disappointing",
        "Outstanding fantastic wonderful",
    ]
    training_labels = [1, 0, 1, 0, 1]

    # Fit the sklearn pipeline directly (outside CIAF)
    sklearn_pipeline.fit(training_texts, training_labels)
    print("✅ Sklearn pipeline fitted successfully")

    # Now create CIAF training data format
    ciaf_training_data = [
        {"content": text, "metadata": {"id": f"{i}", "target": label}}
        for i, (text, label) in enumerate(zip(training_texts, training_labels))
    ]

    # Wrap the trained model
    wrapper = CIAFModelWrapper(
        model=sklearn_pipeline, model_name="SklearnTextClassifier", enable_chaining=True
    )

    # Train with CIAF (will skip model fitting since already trained)
    snapshot = wrapper.train(
        dataset_id="sklearn_text_dataset",
        training_data=ciaf_training_data,
        master_password="test_password",
        model_version="1.0.0",
        fit_model=False,  # Skip fitting since we already trained
    )

    # Test prediction
    test_text = "This is excellent quality"
    prediction, receipt = wrapper.predict(test_text)
    print(f"✅ Prediction: {prediction}")
    print(f"   Receipt: {receipt.receipt_hash[:16]}...")

    # Verify
    verification = wrapper.verify(receipt)
    print(f"✅ Verification: {verification['receipt_integrity']}")

    return True


def test_numerical_sklearn():
    """Test sklearn model with numerical data."""
    print("\n🔢 Testing Sklearn Model with Numerical Data")
    print("=" * 60)

    import json

    from sklearn.linear_model import LinearRegression

    # Create numerical training data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([3, 5, 7, 9, 11])  # y = x1 + x2

    # Create a wrapper class that handles numerical data conversion
    class SklearnNumericalAdapter:
        def __init__(self, sklearn_model):
            self.sklearn_model = sklearn_model

        def fit(self, X_train, y_train):
            # Convert CIAF string format back to numerical for sklearn
            if isinstance(X_train[0], str):
                X_numerical = np.array([json.loads(x) for x in X_train])
            else:
                X_numerical = np.array(X_train)
            return self.sklearn_model.fit(X_numerical, y_train)

        def predict(self, X_test):
            # Handle single prediction (string from CIAF) or batch
            if isinstance(X_test, str):
                X_numerical = np.array([json.loads(X_test)])
            elif isinstance(X_test, list) and len(X_test) > 0:
                if isinstance(X_test[0], str):
                    X_numerical = np.array([json.loads(x) for x in X_test])
                else:
                    X_numerical = np.array([X_test])
            else:
                X_numerical = np.array(X_test)

            return self.sklearn_model.predict(X_numerical)

    # Create CIAF training data format (convert numerical data to JSON strings)
    ciaf_training_data = [
        {
            "content": json.dumps(X[i].tolist()),
            "metadata": {"id": f"{i}", "target": y[i]},
        }
        for i in range(len(X))
    ]

    # Create adapted model
    sklearn_model = LinearRegression()
    adapted_model = SklearnNumericalAdapter(sklearn_model)

    wrapper = CIAFModelWrapper(
        model=adapted_model, model_name="SklearnRegression", enable_chaining=True
    )

    # Train
    snapshot = wrapper.train(
        dataset_id="numerical_regression_dataset",
        training_data=ciaf_training_data,
        master_password="test_password",
        model_version="1.0.0",
    )

    # Predict (convert to JSON string for CIAF compatibility)
    test_input = [3.5, 4.5]  # Expected output: ~8
    prediction, receipt = wrapper.predict(json.dumps(test_input))
    print(f"✅ Prediction for {test_input}: {prediction}")
    print(f"   Receipt: {receipt.receipt_hash[:16]}...")

    return True


def demonstrate_compatibility_patterns():
    """Show different patterns for ML model compatibility."""
    print("\n📋 ML Model Compatibility Patterns with CIAF")
    print("=" * 60)

    patterns = [
        {
            "name": "✅ Custom Models with fit/predict",
            "description": "Models that implement fit() and predict() methods work directly",
        },
        {
            "name": "✅ Pre-trained Models",
            "description": "Use fit_model=False to skip training, just wrap for inference tracking",
        },
        {
            "name": "✅ Sklearn with Numerical Data",
            "description": "Works when data is already in numerical format",
        },
        {
            "name": "✅ Sklearn Pipelines",
            "description": "Pre-trained pipelines work well for inference tracking",
        },
        {
            "name": "⚠️ Sklearn with Raw Text",
            "description": "Needs preprocessing (TfidfVectorizer, etc.) before CIAF wrapper",
        },
        {
            "name": "⚠️ Deep Learning Models",
            "description": "May need tensor conversion and special handling",
        },
        {
            "name": "🔧 Callable Functions",
            "description": "Functions with predict() method work for inference tracking",
        },
    ]

    for pattern in patterns:
        print(f"{pattern['name']}")
        print(f"   {pattern['description']}")
        print()


def main():
    """Run comprehensive sklearn compatibility tests."""
    print("🧪 CIAF + Scikit-Learn Compatibility Examples")
    print("=" * 70)

    results = []

    try:
        results.append(
            ("Sklearn Text (Preprocessed)", test_sklearn_with_preprocessing())
        )
    except Exception as e:
        print(f"❌ Sklearn text failed: {e}")
        results.append(("Sklearn Text (Preprocessed)", False))

    try:
        results.append(("Sklearn Numerical", test_numerical_sklearn()))
    except Exception as e:
        print(f"❌ Sklearn numerical failed: {e}")
        results.append(("Sklearn Numerical", False))

    demonstrate_compatibility_patterns()

    # Summary
    print("=" * 70)
    print("📊 SKLEARN COMPATIBILITY RESULTS")
    print("=" * 70)

    for name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{name:30} : {status}")

    successful = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\nResults: {successful}/{total} sklearn examples worked")

    print("\n" + "=" * 70)
    print("🎯 RECOMMENDATIONS FOR SKLEARN + CIAF")
    print("=" * 70)
    print("1. For text data: Use sklearn pipelines with TfidfVectorizer")
    print("2. For numerical data: Direct integration works well")
    print("3. For pre-trained models: Use fit_model=False")
    print("4. Always test with small datasets first")
    print("5. CIAF provides full provenance even when model training is skipped")
    print("\n🚀 The wrapper is designed to be flexible and gracefully handle failures!")


if __name__ == "__main__":
    main()
