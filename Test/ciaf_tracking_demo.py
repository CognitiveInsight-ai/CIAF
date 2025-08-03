#!/usr/bin/env python3
"""
CIAF Enhanced Training/Testing Data Tracking Demonstration

This script demonstrates how the CIAF framework properly tracks training and test data
with clear separation and audit trails for compliance purposes.

Key Features:
- Training data tracked as: "{data_file_name}_Train"
- Test data tracked as: "{data_file_name}_Test" 
- Test inference tracked through CIAF framework
- Live inference tracked as: "{data_file_name}_Inference"
- Complete audit trail for compliance
"""

import os
import sys
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from job_classifier_model import JobClassifierModel
import numpy as np

def demonstrate_ciaf_data_tracking():
    """
    Demonstrate comprehensive CIAF data tracking for training, testing, and inference
    """
    print("🎯 CIAF Data Tracking Demonstration")
    print("=" * 60)
    print("📋 This demo shows how CIAF tracks all data with proper audit trails:")
    print("   • Training Data: {dataset_name}_Train") 
    print("   • Test Data: {dataset_name}_Test")
    print("   • Test Inference: Tracked through CIAF framework")
    print("   • Live Inference: {dataset_name}_Inference")
    print("=" * 60)
    
    # Initialize model
    print("\n🚀 Step 1: Initialize Job Classifier with CIAF Integration")
    model = JobClassifierModel(model_id="CIAF_Demo_JobClassifier_v1.0")
    
    # Initialize data tracking with specific dataset name
    dataset_name = "hiring_audit_dataset_2025"
    print(f"\n📦 Step 2: Initialize CIAF Data Tracking for '{dataset_name}'")
    model.initialize_data_tracking(dataset_name)
    
    # Generate and prepare data
    print(f"\n📊 Step 3: Generate Sample Data")
    data = model.generate_sample_data(n_samples=1000)
    X, y, data_processed = model.preprocess_data(data)
    print(f"✅ Generated {len(data)} candidate records")
    
    # Train model with CIAF tracking
    print(f"\n🤖 Step 4: Train Model with CIAF Data Tracking")
    print(f"   📁 Training data will be tracked as: '{dataset_name}_Train'")
    print(f"   📁 Test data will be tracked as: '{dataset_name}_Test'")
    print(f"   🧪 Test inference will be tracked through CIAF framework")
    
    X_test, y_test, y_pred = model.train_model(X, y)
    
    # Evaluate model
    print(f"\n⚖️ Step 5: Evaluate Model Bias and Fairness")
    bias_results, fairness_results = model.evaluate_bias(X_test, y_test, y_pred, data_processed)
    compliance_score = model.calculate_compliance_score()
    print(f"📋 Compliance Score: {compliance_score:.2%}")
    
    # Demonstrate live inference tracking
    print(f"\n🎯 Step 6: Live Inference with CIAF Tracking")
    print(f"   📁 Live predictions will be tracked as: '{dataset_name}_Inference'")
    
    # Simulate multiple inference requests
    for batch_num in range(3):
        print(f"\n   🔄 Inference Batch {batch_num + 1}:")
        
        # Create sample inference data with proper feature structure
        # Generate realistic candidate data instead of random values
        sample_data = []
        for i in range(2):  # 2 candidates per batch
            candidate = {
                'education_score': np.random.uniform(60, 100),
                'experience_years': np.random.uniform(0, 20),
                'skill_score': np.random.uniform(50, 100),
                'age': np.random.uniform(22, 65),
                'gender_encoded': np.random.choice([0, 1, 2])  # F, M, NB
            }
            sample_data.append(candidate)
        
        # Convert to DataFrame with proper column names to match training
        sample_df = pd.DataFrame(sample_data)
        sample_candidates = model.scaler.transform(sample_df)
        
        # Make predictions (automatically tracked through CIAF)
        predictions, uncertainties = model.predict_with_uncertainty(sample_candidates)
        
        for i, (pred, unc) in enumerate(zip(predictions, uncertainties)):
            hire_decision = "HIRE" if pred[1] > 0.5 else "NO HIRE"
            print(f"     Candidate {batch_num*2 + i + 1}: {hire_decision} (confidence: {pred[1]:.3f}, uncertainty: {unc:.3f})")
    
    # Display comprehensive tracking summary
    print(f"\n📊 Step 7: CIAF Tracking Summary")
    model_info = model.get_model_info()
    
    print(f"\n📁 Data File Tracking:")
    if 'training_data_tracking' in model_info:
        train_info = model_info['training_data_tracking']
        print(f"   🏋️ Training: {train_info['data_file_name']} - {train_info['total_items']} samples tracked")
    
    if 'test_data_tracking' in model_info:
        test_info = model_info['test_data_tracking']
        print(f"   🧪 Testing: {test_info['data_file_name']} - {test_info['total_items']} samples tracked")
    
    if 'inference_data_tracking' in model_info:
        inference_info = model_info['inference_data_tracking']
        print(f"   🎯 Inference: {inference_info['data_file_name']} - {inference_info['total_items']} predictions tracked")
    
    # Show audit trail
    print(f"\n🔍 Step 8: Generate Audit Trail for Compliance")
    audit_trail = model.get_ciaf_audit_trail()
    
    print(f"\n📋 Compliance Audit Report:")
    print(f"   Model ID: {audit_trail['model_id']}")
    print(f"   Base Dataset: {audit_trail['data_file_base_name']}")
    print(f"   Audit Timestamp: {audit_trail['audit_timestamp']}")
    print(f"   Compliance Score: {audit_trail['compliance_metrics']['overall_compliance_score']:.2%}")
    
    print(f"\n📁 Complete Data Lineage:")
    for phase, lineage in audit_trail['data_lineage'].items():
        print(f"   {phase.upper()}:")
        print(f"     📄 Data File: {lineage['data_file']}")
        
        if 'total_samples' in lineage:
            print(f"     📊 Total Samples: {lineage['total_samples']}")
        if 'total_predictions' in lineage:
            print(f"     📊 Total Predictions: {lineage['total_predictions']}")
        
        integrity_status = "✅ Verified" if lineage['data_integrity_verified'] else "❌ Failed"
        print(f"     🔒 Data Integrity: {integrity_status}")
        
        if 'inference_results_tracked' in lineage:
            inference_status = "✅ Complete" if lineage['inference_results_tracked'] else "❌ Missing"
            print(f"     🧪 Inference Tracking: {inference_status}")
    
    print(f"\n✅ CIAF Data Tracking Complete!")
    print(f"🎯 Key Benefits for Audit:")
    print(f"   • Clear separation of training vs test vs inference data")
    print(f"   • Unique file names for each data type")
    print(f"   • Complete cryptographic audit trail")
    print(f"   • Metadata preserved for each phase")
    print(f"   • Compliance scores tracked throughout")
    
    return model

if __name__ == "__main__":
    model = demonstrate_ciaf_data_tracking()
