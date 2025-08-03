#!/usr/bin/env python3
"""
Quick test script to verify models can be loaded and work
"""

import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

def test_job_classifier():
    """Test job classifier model"""
    print("Testing Job Classifier Model...")
    try:
        from models.job_classifier_model import JobClassifierModel
        
        # Create and test model
        model = JobClassifierModel()
        print("✅ Job classifier model created successfully")
        
        # Generate sample data
        data = model.generate_sample_data(n_samples=100)
        print(f"✅ Generated {len(data)} sample records")
        
        # Preprocess data
        X, y, data_processed = model.preprocess_data(data)
        print(f"✅ Data preprocessed: {X.shape}")
        
        # Train model
        X_test, y_test, y_pred = model.train_model(X, y)
        print("✅ Model trained successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Job classifier test failed: {e}")
        return False

def test_credit_scoring():
    """Test credit scoring model"""
    print("\nTesting Credit Scoring Model...")
    try:
        from models.credit_scoring_model import CreditScoringModel
        
        # Create and test model
        model = CreditScoringModel()
        print("✅ Credit scoring model created successfully")
        
        # Generate sample data
        data = model.generate_sample_credit_data(n_samples=100)
        print(f"✅ Generated {len(data)} sample records")
        
        # Preprocess data
        X, y, data_processed = model.preprocess_credit_data(data)
        print(f"✅ Data preprocessed: {X.shape}")
        
        # Train model
        X_test, y_test, y_pred, y_prob = model.train_credit_model(X, y)
        print("✅ Model trained successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Credit scoring test failed: {e}")
        return False

def test_ct_scan():
    """Test CT scan model"""
    print("\nTesting CT Scan Model...")
    try:
        from models.ct_scan_model import CTScanModel
        
        # Create and test model
        model = CTScanModel()
        print("✅ CT scan model created successfully")
        
        # Generate sample data
        data = model.generate_sample_ct_data(n_samples=100)
        print(f"✅ Generated {len(data)} sample records")
        
        # Preprocess data
        X, y, data_processed = model.preprocess_medical_data(data)
        print(f"✅ Data preprocessed: {X.shape}")
        
        # Train model
        X_test, y_test, y_pred, y_prob = model.train_medical_model(X, y)
        print("✅ Model trained successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ CT scan test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 CIAF Model Testing Script")
    print("=" * 50)
    
    results = []
    
    # Test all models
    results.append(test_job_classifier())
    results.append(test_credit_scoring())
    results.append(test_ct_scan())
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"✅ Passed: {sum(results)}/3")
    print(f"❌ Failed: {3 - sum(results)}/3")
    
    if all(results):
        print("🎉 All models working correctly!")
    else:
        print("⚠️  Some models have issues that need to be fixed.")
