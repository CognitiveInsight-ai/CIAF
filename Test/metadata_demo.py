#!/usr/bin/env python3
"""
CIAF Metadata Storage Demonstration

This script demonstrates how to use the CIAF metadata storage system
to save, retrieve, and manage metadata throughout the AI pipeline.
"""

import sys
import os
import json
from datetime import datetime

# Add CIAF to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from ciaf.metadata_storage import MetadataStorage, get_metadata_storage
    from ciaf.metadata_config import MetadataConfig, get_metadata_config, create_config_template
    from ciaf.metadata_integration import (
        ModelMetadataManager, ComplianceTracker, MetadataCapture, 
        capture_metadata, create_model_manager
    )
    METADATA_AVAILABLE = True
    print("✅ CIAF metadata storage components imported successfully")
except ImportError as e:
    print(f"❌ Error importing CIAF metadata components: {e}")
    METADATA_AVAILABLE = False
    sys.exit(1)

def demonstrate_basic_metadata_storage():
    """Demonstrate basic metadata storage operations."""
    print("\n" + "="*60)
    print("🔹 BASIC METADATA STORAGE DEMONSTRATION")
    print("="*60)
    
    # Initialize metadata storage with JSON backend
    storage = MetadataStorage("demo_metadata", backend="json")
    
    # Save some example metadata
    print("\n📝 Saving metadata...")
    
    # Data ingestion metadata
    data_id = storage.save_metadata(
        model_name="demo_classifier",
        stage="data_ingestion",
        event_type="data_loaded",
        metadata={
            "dataset_name": "customer_data.csv",
            "rows": 10000,
            "columns": 15,
            "missing_values": 25,
            "data_quality_score": 0.92,
            "source": "production_database",
            "schema_version": "v2.1"
        },
        details="Successfully loaded customer dataset for training"
    )
    print(f"✅ Data ingestion metadata saved with ID: {data_id[:8]}...")
    
    # Training metadata
    training_id = storage.save_metadata(
        model_name="demo_classifier",
        stage="training",
        event_type="model_trained",
        metadata={
            "algorithm": "RandomForestClassifier",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2
            },
            "training_samples": 8000,
            "validation_samples": 2000,
            "accuracy": 0.94,
            "precision": 0.92,
            "recall": 0.96,
            "f1_score": 0.94,
            "training_time_seconds": 45.6
        },
        details="Model training completed successfully"
    )
    print(f"✅ Training metadata saved with ID: {training_id[:8]}...")
    
    # Inference metadata
    inference_id = storage.save_metadata(
        model_name="demo_classifier",
        stage="inference",
        event_type="prediction_made",
        metadata={
            "input_samples": 1,
            "prediction": "positive",
            "confidence": 0.87,
            "processing_time_ms": 12.5,
            "model_version": "v1.2.0"
        },
        details="Single prediction made for customer ID 12345"
    )
    print(f"✅ Inference metadata saved with ID: {inference_id[:8]}...")
    
    # Retrieve metadata
    print("\n📖 Retrieving metadata...")
    
    # Get specific metadata by ID
    retrieved_data = storage.get_metadata(data_id)
    if retrieved_data:
        print(f"✅ Retrieved data ingestion metadata:")
        print(f"   📊 Dataset: {retrieved_data['metadata']['dataset_name']}")
        print(f"   📈 Quality Score: {retrieved_data['metadata']['data_quality_score']}")
    
    # Get all model metadata
    all_metadata = storage.get_model_metadata("demo_classifier", limit=10)
    print(f"\n✅ Found {len(all_metadata)} metadata records for demo_classifier")
    
    # Get pipeline trace
    pipeline_trace = storage.get_pipeline_trace("demo_classifier")
    print(f"✅ Pipeline trace includes {len(pipeline_trace['stages'])} stages:")
    for stage, events in pipeline_trace['stages'].items():
        print(f"   🔸 {stage}: {len(events)} events")
    
    return storage

def demonstrate_model_metadata_manager():
    """Demonstrate the ModelMetadataManager for structured metadata logging."""
    print("\n" + "="*60)
    print("🔹 MODEL METADATA MANAGER DEMONSTRATION")
    print("="*60)
    
    # Create model metadata manager
    manager = create_model_manager("advanced_classifier", "2.1.0")
    print("✅ Created ModelMetadataManager for advanced_classifier v2.1.0")
    
    # Log data ingestion
    print("\n📊 Logging data ingestion...")
    data_info = {
        "source": "customer_feedback.parquet",
        "rows": 50000,
        "columns": 25,
        "categorical_features": 8,
        "numerical_features": 17,
        "missing_percentage": 0.03,
        "data_drift_score": 0.02,
        "quality_checks_passed": 12,
        "quality_checks_failed": 0
    }
    data_id = manager.log_data_ingestion(data_info, "High-quality customer feedback data ingested")
    print(f"✅ Data ingestion logged with ID: {data_id[:8]}...")
    
    # Log preprocessing
    print("\n🔧 Logging data preprocessing...")
    preprocessing_info = {
        "steps": ["missing_value_imputation", "categorical_encoding", "feature_scaling"],
        "imputation_strategy": "median",
        "encoding_method": "one_hot",
        "scaler_type": "StandardScaler",
        "features_dropped": 2,
        "features_created": 5,
        "final_feature_count": 28,
        "preprocessing_time_seconds": 23.4
    }
    preprocess_id = manager.log_data_preprocessing(preprocessing_info, "Data preprocessing completed")
    print(f"✅ Preprocessing logged with ID: {preprocess_id[:8]}...")
    
    # Log training
    print("\n🎯 Logging model training...")
    training_config = {
        "algorithm": "GradientBoostingClassifier",
        "hyperparameters": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 8,
            "subsample": 0.8
        },
        "cross_validation_folds": 5,
        "hyperparameter_tuning": "grid_search",
        "early_stopping": True
    }
    training_start_id = manager.log_training_start(training_config, "Starting training with hyperparameter tuning")
    print(f"✅ Training start logged with ID: {training_start_id[:8]}...")
    
    # Simulate training completion
    training_results = {
        "final_accuracy": 0.96,
        "cv_accuracy_mean": 0.94,
        "cv_accuracy_std": 0.02,
        "precision": 0.95,
        "recall": 0.97,
        "f1_score": 0.96,
        "auc_roc": 0.98,
        "training_time_minutes": 12.3,
        "best_hyperparameters": {
            "n_estimators": 150,
            "learning_rate": 0.12,
            "max_depth": 6
        },
        "feature_importance_top_5": [
            {"feature": "customer_satisfaction", "importance": 0.23},
            {"feature": "purchase_frequency", "importance": 0.19},
            {"feature": "support_interactions", "importance": 0.15},
            {"feature": "account_age_days", "importance": 0.12},
            {"feature": "total_spent", "importance": 0.10}
        ]
    }
    training_complete_id = manager.log_training_complete(training_results, "Training completed successfully")
    print(f"✅ Training completion logged with ID: {training_complete_id[:8]}...")
    
    # Log validation
    print("\n✅ Logging model validation...")
    validation_results = {
        "test_accuracy": 0.955,
        "test_precision": 0.942,
        "test_recall": 0.968,
        "test_f1": 0.955,
        "confusion_matrix": [[450, 25], [18, 507]],
        "classification_report": {
            "class_0": {"precision": 0.96, "recall": 0.95, "f1-score": 0.95},
            "class_1": {"precision": 0.95, "recall": 0.97, "f1-score": 0.96}
        },
        "validation_dataset_size": 1000,
        "validation_method": "holdout_test_set"
    }
    validation_id = manager.log_validation(validation_results, "Model validation on holdout test set")
    print(f"✅ Validation logged with ID: {validation_id[:8]}...")
    
    # Get pipeline trace
    print("\n📋 Retrieving pipeline trace...")
    trace = manager.get_pipeline_trace()
    print(f"✅ Pipeline trace contains {len(trace['stages'])} stages:")
    for stage, events in trace['stages'].items():
        latest_event = max(events, key=lambda x: x['timestamp'])
        print(f"   🔸 {stage}: {len(events)} events (latest: {latest_event['event_type']})")
    
    return manager

def demonstrate_compliance_tracking():
    """Demonstrate compliance tracking across different frameworks."""
    print("\n" + "="*60)
    print("🔹 COMPLIANCE TRACKING DEMONSTRATION")
    print("="*60)
    
    # Create compliance tracker
    manager = create_model_manager("compliance_model", "1.0.0")
    tracker = ComplianceTracker(manager)
    print("✅ Created ComplianceTracker for compliance_model")
    
    # Track GDPR compliance
    print("\n🛡️ Tracking GDPR compliance...")
    gdpr_id = tracker.track_gdpr_compliance(
        data_protection_measures={
            "encryption": True,
            "anonymization": True,
            "access_controls": True,
            "data_minimization": True,
            "retention_policy": True
        },
        consent_management={
            "explicit_consent": True,
            "withdrawal_mechanism": True,
            "consent_logging": True,
            "granular_consent": True
        },
        right_to_explanation=True
    )
    print(f"✅ GDPR compliance tracked with ID: {gdpr_id[:8]}...")
    print(f"   📊 GDPR Score: {tracker.compliance_scores.get('GDPR', 0):.3f}")
    
    # Track FDA compliance (for medical AI)
    print("\n🏥 Tracking FDA compliance...")
    fda_id = tracker.track_fda_compliance(
        clinical_validation={
            "clinical_studies": True,
            "performance_validation": True,
            "clinical_trials": False,  # Not yet completed
            "regulatory_submission": False
        },
        safety_measures={
            "risk_assessment": True,
            "monitoring_plan": True,
            "adverse_event_reporting": True,
            "safety_updates": True
        },
        quality_management={
            "iso_13485": True,
            "documentation": True,
            "change_control": True,
            "validation_procedures": True
        }
    )
    print(f"✅ FDA compliance tracked with ID: {fda_id[:8]}...")
    print(f"   📊 FDA Score: {tracker.compliance_scores.get('FDA', 0):.3f}")
    
    # Track EEOC compliance (for hiring AI)
    print("\n👥 Tracking EEOC compliance...")
    eeoc_id = tracker.track_eeoc_compliance(
        bias_assessment={
            "disparate_impact": 0.15,  # Low disparate impact is good
            "statistical_parity": 0.88,  # High statistical parity is good
            "individual_fairness": 0.92
        },
        fairness_metrics={
            "equalized_odds": 0.89,
            "calibration": 0.91,
            "demographic_parity": 0.85
        },
        protected_classes=["race", "gender", "age", "disability"]
    )
    print(f"✅ EEOC compliance tracked with ID: {eeoc_id[:8]}...")
    print(f"   📊 EEOC Score: {tracker.compliance_scores.get('EEOC', 0):.3f}")
    
    # Get overall compliance score
    overall_score = tracker.get_overall_compliance_score()
    print(f"\n🎯 Overall Compliance Score: {overall_score:.3f}")
    
    return tracker

def demonstrate_metadata_capture_decorator():
    """Demonstrate the @capture_metadata decorator."""
    print("\n" + "="*60)
    print("🔹 METADATA CAPTURE DECORATOR DEMONSTRATION")
    print("="*60)
    
    # Define functions with metadata capture
    @capture_metadata("decorator_model", "data_processing", "data_cleaning")
    def clean_data(data_path: str, remove_outliers: bool = True):
        """Clean and preprocess data."""
        import time
        import random
        
        print(f"   🧹 Cleaning data from {data_path}")
        time.sleep(0.5)  # Simulate processing
        
        rows_cleaned = random.randint(100, 1000)
        outliers_removed = random.randint(5, 50) if remove_outliers else 0
        
        return {
            "rows_processed": rows_cleaned,
            "outliers_removed": outliers_removed,
            "quality_score": 0.93
        }
    
    @capture_metadata("decorator_model", "feature_engineering", "feature_creation", 
                     capture_result=True, capture_performance=True)
    def create_features(base_features: int, interaction_terms: bool = False):
        """Create new features from existing ones."""
        import time
        import random
        
        print(f"   🔧 Creating features from {base_features} base features")
        time.sleep(0.3)  # Simulate processing
        
        new_features = random.randint(5, 15)
        if interaction_terms:
            new_features += random.randint(10, 25)
        
        return {
            "new_features_created": new_features,
            "total_features": base_features + new_features,
            "feature_types": ["numerical", "categorical", "interaction"]
        }
    
    @capture_metadata("decorator_model", "model_evaluation", "performance_assessment")
    def evaluate_model(model_type: str, test_size: int):
        """Evaluate model performance."""
        import time
        import random
        
        print(f"   📊 Evaluating {model_type} model on {test_size} samples")
        time.sleep(0.4)  # Simulate evaluation
        
        return {
            "accuracy": round(random.uniform(0.85, 0.98), 3),
            "precision": round(random.uniform(0.80, 0.95), 3),
            "recall": round(random.uniform(0.82, 0.96), 3)
        }
    
    # Execute functions with automatic metadata capture
    print("\n🚀 Executing functions with metadata capture...")
    
    clean_result = clean_data("/data/customer_data.csv", remove_outliers=True)
    print(f"✅ Data cleaning completed: {clean_result['rows_processed']} rows processed")
    
    feature_result = create_features(20, interaction_terms=True)
    print(f"✅ Feature creation completed: {feature_result['new_features_created']} new features")
    
    eval_result = evaluate_model("RandomForest", 1000)
    print(f"✅ Model evaluation completed: {eval_result['accuracy']} accuracy")
    
    # Check metadata was captured
    storage = get_metadata_storage()
    decorator_metadata = storage.get_model_metadata("decorator_model", limit=10)
    print(f"\n📊 Captured {len(decorator_metadata)} metadata records automatically:")
    for record in decorator_metadata:
        print(f"   🔸 {record['stage']}.{record['event_type']} at {record['timestamp'][:19]}")

def demonstrate_context_manager():
    """Demonstrate using MetadataCapture as a context manager."""
    print("\n" + "="*60)
    print("🔹 METADATA CONTEXT MANAGER DEMONSTRATION")
    print("="*60)
    
    # Use context manager for complex operations
    print("\n🔄 Using context manager for batch processing...")
    
    with MetadataCapture("context_model", "batch_processing", "data_pipeline") as capture:
        capture.add_metadata("batch_id", "batch_2024_001")
        capture.add_metadata("input_source", "production_database")
        
        # Simulate batch processing steps
        import time
        import random
        
        # Step 1: Data extraction
        print("   📥 Extracting data...")
        time.sleep(0.2)
        rows_extracted = random.randint(5000, 15000)
        capture.add_metadata("rows_extracted", rows_extracted)
        
        # Step 2: Data transformation
        print("   🔄 Transforming data...")
        time.sleep(0.3)
        transformations_applied = ["normalize", "encode_categorical", "handle_missing"]
        capture.add_metadata("transformations", transformations_applied)
        
        # Step 3: Data validation
        print("   ✅ Validating data...")
        time.sleep(0.1)
        validation_passed = True
        capture.add_metadata("validation_passed", validation_passed)
        
        # Step 4: Data export
        print("   📤 Exporting processed data...")
        time.sleep(0.2)
        output_file = f"processed_batch_{random.randint(1000,9999)}.parquet"
        capture.add_metadata("output_file", output_file)
        
        print(f"✅ Batch processing completed: {rows_extracted} rows processed")
    
    print("✅ Metadata automatically saved when exiting context manager")

def demonstrate_metadata_export():
    """Demonstrate metadata export capabilities."""
    print("\n" + "="*60)
    print("🔹 METADATA EXPORT DEMONSTRATION")
    print("="*60)
    
    storage = get_metadata_storage()
    
    # Export specific model metadata
    print("\n📁 Exporting model metadata...")
    
    # JSON export
    json_path = storage.export_metadata("demo_classifier", "json")
    print(f"✅ JSON export: {json_path}")
    
    # CSV export
    csv_path = storage.export_metadata("demo_classifier", "csv")
    print(f"✅ CSV export: {csv_path}")
    
    # XML export
    xml_path = storage.export_metadata("demo_classifier", "xml")
    print(f"✅ XML export: {xml_path}")
    
    # Export all metadata
    all_json_path = storage.export_metadata(format="json")
    print(f"✅ All metadata JSON export: {all_json_path}")

def demonstrate_configuration():
    """Demonstrate metadata configuration management."""
    print("\n" + "="*60)
    print("🔹 CONFIGURATION DEMONSTRATION")
    print("="*60)
    
    # Create configuration templates
    print("\n⚙️ Creating configuration templates...")
    
    templates = ["development", "production", "testing", "high_performance"]
    for template in templates:
        try:
            config_path = create_config_template(template, f"demo_config_{template}.json")
            print(f"✅ Created {template} template: {config_path}")
        except Exception as e:
            print(f"❌ Error creating {template} template: {e}")
    
    # Show current configuration
    config = get_metadata_config()
    print(f"\n📋 Current configuration:")
    print(f"   🔸 Storage backend: {config.get('storage_backend')}")
    print(f"   🔸 Storage path: {config.get('storage_path')}")
    print(f"   🔸 Retention days: {config.get('metadata_retention_days')}")
    print(f"   🔸 Compliance frameworks: {len(config.get('compliance_frameworks', []))}")
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print(f"❌ Configuration errors: {errors}")
    else:
        print("✅ Configuration is valid")

def main():
    """Main demonstration function."""
    print("🚀 CIAF METADATA STORAGE SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    if not METADATA_AVAILABLE:
        print("❌ Metadata components not available. Exiting.")
        return
    
    try:
        # Run all demonstrations
        demonstrate_basic_metadata_storage()
        demonstrate_model_metadata_manager()
        demonstrate_compliance_tracking()
        demonstrate_metadata_capture_decorator()
        demonstrate_context_manager()
        demonstrate_metadata_export()
        demonstrate_configuration()
        
        print("\n" + "="*80)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\n📋 SUMMARY:")
        print("✅ Basic metadata storage operations")
        print("✅ Structured model metadata management")
        print("✅ Compliance tracking across frameworks")
        print("✅ Automatic metadata capture with decorators")
        print("✅ Context manager for complex operations")
        print("✅ Metadata export in multiple formats")
        print("✅ Configuration management")
        
        print("\n💡 Next Steps:")
        print("1. Integrate metadata storage into your existing models")
        print("2. Configure appropriate retention policies")
        print("3. Set up compliance monitoring for your use case")
        print("4. Export metadata for regulatory reporting")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
