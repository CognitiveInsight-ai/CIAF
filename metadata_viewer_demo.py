#!/usr/bin/env python3
"""
CIAF Metadata Viewer Demonstration

This script demonstrates the complete metadata storage and visualization system,
including saving metadata and viewing it through the web dashboard.
"""

import sys
import os
import json
from datetime import datetime, timedelta
import random

# Add CIAF to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from ciaf.metadata_storage import MetadataStorage
    from ciaf.metadata_integration import ModelMetadataManager, ComplianceTracker
    from ciaf.metadata_config import MetadataConfig
except ImportError as e:
    print(f"Error importing CIAF modules: {e}")
    print("Please ensure CIAF is properly installed")
    sys.exit(1)


def generate_sample_metadata():
    """Generate sample metadata for demonstration."""
    print("🔧 Generating sample metadata...")
    
    # Initialize metadata storage
    storage = MetadataStorage(storage_path="ciaf_metadata_demo", backend="json")
    
    # Sample models and frameworks
    models = [
        ("job_classifier_v1", "EEOC"),
        ("ct_scan_classifier_v2", "FDA"), 
        ("credit_scoring_v3", "FCRA"),
        ("loan_approval_v1", "FCRA"),
        ("medical_diagnosis_v2", "HIPAA"),
        ("hiring_assistant_v3", "EEOC")
    ]
    
    stages = ["data_collection", "preprocessing", "training", "validation", "testing", "production"]
    event_types = ["model_training", "data_validation", "compliance_check", "performance_evaluation"]
    
    # Generate metadata entries for the past 30 days
    for i in range(50):  # Generate 50 sample entries
        model_name, framework = random.choice(models)
        stage = random.choice(stages)
        event_type = random.choice(event_types)
        
        # Random timestamp within last 30 days
        days_ago = random.randint(0, 30)
        timestamp = datetime.now() - timedelta(days=days_ago)
        
        # Generate realistic metadata based on model type
        if "job_classifier" in model_name:
            metadata = {
                "dataset_size": random.randint(5000, 50000),
                "features_used": ["education", "experience", "skills", "certifications"],
                "accuracy": round(random.uniform(0.78, 0.94), 3),
                "bias_metrics": {
                    "demographic_parity": round(random.uniform(0.75, 0.95), 3),
                    "equal_opportunity": round(random.uniform(0.80, 0.96), 3),
                    "statistical_parity": round(random.uniform(0.72, 0.92), 3)
                },
                "protected_groups": ["gender", "race", "age"],
                "compliance_score": round(random.uniform(0.75, 0.95), 3)
            }
        elif "ct_scan" in model_name:
            metadata = {
                "dataset_size": random.randint(1000, 10000),
                "image_resolution": "512x512",
                "model_architecture": "ResNet50",
                "accuracy": round(random.uniform(0.85, 0.98), 3),
                "sensitivity": round(random.uniform(0.82, 0.96), 3),
                "specificity": round(random.uniform(0.88, 0.99), 3),
                "auc_score": round(random.uniform(0.90, 0.99), 3),
                "validation_set_size": random.randint(200, 2000),
                "compliance_score": round(random.uniform(0.85, 0.98), 3)
            }
        elif "credit_scoring" in model_name:
            metadata = {
                "dataset_size": random.randint(10000, 100000),
                "features_used": ["income", "credit_history", "employment_status", "debt_ratio"],
                "accuracy": round(random.uniform(0.72, 0.88), 3),
                "precision": round(random.uniform(0.70, 0.85), 3),
                "recall": round(random.uniform(0.68, 0.82), 3),
                "fairness_metrics": {
                    "statistical_parity": round(random.uniform(0.65, 0.85), 3),
                    "individual_fairness": round(random.uniform(0.70, 0.88), 3),
                    "calibration": round(random.uniform(0.75, 0.90), 3)
                },
                "default_rate": round(random.uniform(0.05, 0.15), 3),
                "compliance_score": round(random.uniform(0.68, 0.88), 3)
            }
        else:
            # Generic metadata
            metadata = {
                "dataset_size": random.randint(1000, 50000),
                "accuracy": round(random.uniform(0.70, 0.95), 3),
                "performance_metrics": {
                    "precision": round(random.uniform(0.70, 0.90), 3),
                    "recall": round(random.uniform(0.65, 0.85), 3),
                    "f1_score": round(random.uniform(0.68, 0.87), 3)
                },
                "compliance_score": round(random.uniform(0.70, 0.90), 3)
            }
        
        # Add common metadata fields
        metadata.update({
            "timestamp": timestamp.isoformat(),
            "pipeline_id": f"pipeline_{random.randint(1000, 9999)}",
            "user_id": f"user_{random.randint(100, 999)}",
            "environment": random.choice(["development", "staging", "production"]),
            "version": f"v{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
            "compliance_framework": framework
        })
        
        # Save metadata
        metadata_id = storage.save_metadata(
            model_name=model_name,
            stage=stage,
            event_type=event_type,
            metadata=metadata,
            model_version=metadata["version"]
        )
        
        print(f"  ✅ Saved metadata: {metadata_id[:8]}... ({model_name}, {stage})")
    
    print(f"\n📊 Generated 50 sample metadata entries")
    return storage


def demonstrate_metadata_integration():
    """Demonstrate the metadata integration tools."""
    print("\n🔧 Demonstrating metadata integration tools...")
    
    # Initialize metadata manager
    manager = ModelMetadataManager(
        model_name="demo_classifier",
        model_version="v2.1.0"
    )
    
    # Simulate model lifecycle with metadata capture
    print("  📝 Capturing model lifecycle metadata...")
    
    # Data collection stage
    data_collection_id = manager.log_data_ingestion({
        "source": "customer_database",
        "records_count": 15000,
        "data_quality_score": 0.92,
        "privacy_compliance": "GDPR_compliant"
    })
    print(f"    ✅ Data ingestion logged: {data_collection_id[:8]}...")
    
    # Preprocessing stage
    preprocessing_id = manager.log_data_preprocessing({
        "missing_values_handled": True,
        "outliers_removed": 234,
        "feature_engineering": ["age_binning", "income_normalization"],
        "anonymization_applied": True
    })
    print(f"    ✅ Data preprocessing logged: {preprocessing_id[:8]}...")
    
    # Training stage
    training_start_id = manager.log_training_start({
        "algorithm": "RandomForest",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5
        },
        "training_time_minutes": 23,
        "cross_validation_score": 0.87
    })
    print(f"    ✅ Training start logged: {training_start_id[:8]}...")
    
    training_complete_id = manager.log_training_complete({
        "final_accuracy": 0.89,
        "training_loss": 0.12,
        "validation_accuracy": 0.87,
        "training_duration_minutes": 45
    })
    print(f"    ✅ Training completion logged: {training_complete_id[:8]}...")
    
    # Validation stage
    validation_id = manager.log_validation({
        "test_accuracy": 0.85,
        "precision": 0.87,
        "recall": 0.83,
        "f1_score": 0.85,
        "auc_score": 0.91
    })
    print(f"    ✅ Validation logged: {validation_id[:8]}...")
    
    # Production deployment
    deployment_id = manager.log_model_deployment({
        "deployment_environment": "production",
        "model_version": "v2.1.0",
        "rollout_strategy": "blue_green",
        "monitoring_enabled": True
    })
    print(f"    ✅ Deployment logged: {deployment_id[:8]}...")
    
    print("  ✅ Model lifecycle metadata captured")
    
    # Demonstrate compliance tracking
    print("  📋 Demonstrating compliance tracking...")
    
    tracker = ComplianceTracker(model_manager=manager)
    
    # Add compliance events
    gdpr_compliance_id = tracker.track_gdpr_compliance(
        data_protection_measures={
            "encryption": True,
            "anonymization": True,
            "data_minimization": True,
            "retention_policy": "2_years"
        },
        consent_management={
            "consent_obtained": True,
            "consent_type": "explicit",
            "withdrawal_mechanism": True
        },
        right_to_explanation=True
    )
    
    print(f"  ✅ GDPR compliance tracked: {gdpr_compliance_id[:8]}...")
    
    # Track EEOC compliance for bias assessment
    eeoc_compliance_id = tracker.track_eeoc_compliance(
        bias_assessment={
            "bias_testing_performed": True,
            "statistical_significance": True,
            "sample_size_adequate": True
        },
        fairness_metrics={
            "demographic_parity": 0.89,
            "equal_opportunity": 0.91,
            "statistical_parity": 0.87
        },
        protected_classes=["gender", "race", "age"]
    )
    
    print(f"  ✅ EEOC compliance tracked: {eeoc_compliance_id[:8]}...")
    
    print("  ✅ Compliance events tracked")
    
    return manager, tracker


def demonstrate_metadata_export():
    """Demonstrate metadata export functionality."""
    print("\n📤 Demonstrating metadata export...")
    
    storage = MetadataStorage(storage_path="ciaf_metadata_demo", backend="json")
    
    # Export metadata in different formats
    export_formats = ["json", "csv", "xml"]
    
    for format_name in export_formats:
        try:
            export_path = storage.export_metadata(
                model_name="demo_classifier",
                format=format_name
            )
            print(f"  ✅ Exported {format_name.upper()}: {export_path}")
        except Exception as e:
            print(f"  ❌ Failed to export {format_name.upper()}: {e}")
    
    return storage


def demonstrate_web_integration():
    """Demonstrate web dashboard integration."""
    print("\n🌐 Demonstrating web dashboard integration...")
    
    print("  📊 Metadata viewer features:")
    print("    • Multiple view modes (Cards, Table, JSON)")
    print("    • Advanced filtering by date, model, framework")
    print("    • Real-time search across all metadata")
    print("    • Pagination for large datasets")
    print("    • Export functionality (JSON, CSV)")
    print("    • Detailed metadata inspection")
    print("    • Compliance score visualization")
    
    print("\n  🔗 To access the metadata viewer:")
    print("    1. Start the Flask application: python Test/web/app.py")
    print("    2. Open http://localhost:5000 in your browser")
    print("    3. Navigate to the Compliance Dashboard")
    print("    4. Click 'View Metadata' button")
    
    print("\n  🎛️ Available API endpoints:")
    print("    • GET /api/metadata/list - List all metadata")
    print("    • GET /api/metadata/export?format=csv - Export metadata")


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("🚀 CIAF Metadata Viewer Demonstration")
    print("=" * 60)
    
    try:
        # Generate sample data
        storage = generate_sample_metadata()
        
        # Demonstrate integration tools  
        manager, tracker = demonstrate_metadata_integration()
        
        # Demonstrate export functionality
        demonstrate_metadata_export()
        
        # Show web integration info
        demonstrate_web_integration()
        
        print("\n" + "=" * 60)
        print("✅ METADATA SYSTEM DEMONSTRATION COMPLETE")
        print("=" * 60)
        
        print("\n📊 Summary:")
        print(f"  • Generated 50+ sample metadata entries")
        print(f"  • Demonstrated model lifecycle tracking")
        print(f"  • Showed compliance monitoring capabilities")
        print(f"  • Exported metadata in multiple formats")
        print(f"  • Created web-accessible metadata viewer")
        
        print("\n🎯 Next Steps:")
        print("  1. Start the Flask web application")
        print("  2. Access the metadata viewer in your browser")
        print("  3. Explore different view modes and filters") 
        print("  4. Export metadata for external analysis")
        print("  5. Integrate with your existing ML pipelines")
        
        print("\n📁 Files created:")
        print(f"  • Metadata storage: ciaf_metadata_demo/")
        print(f"  • Web dashboard: Test/web/templates/compliance_dashboard.html")
        print(f"  • API endpoints: Test/web/app.py")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
