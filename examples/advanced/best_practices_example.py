#!/usr/bin/env python3
"""
CIAF Best Practices: Pre-Ingestion Validation & Proper Capsulation Tracking

This example demonstrates the recommended workflow for addressing the
train/test split capsulation issue identified by the user.

Key Insights:
• Data capsulation percentages may be <100% due to random train/test splits
• Bias and quality validation must happen BEFORE any data splitting
• Track capsulation by phase (training/testing) separately
• Always validate the complete dataset for representative sampling

Author: CIAF Development Team
Version: 2.1.0
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from ciaf.anchoring.dataset_anchor import DatasetAnchor

# Import CIAF components
from ciaf.compliance.pre_ingestion_validator import PreIngestionValidator
from ciaf.metadata_integration import create_model_manager


def create_sample_dataset(n_samples=1000, add_bias=True):
    """
    Create a sample hiring dataset with optional bias injection.

    Args:
        n_samples: Number of samples to generate
        add_bias: Whether to inject bias into the dataset

    Returns:
        pandas.DataFrame: Generated dataset
    """
    np.random.seed(42)  # For reproducibility

    # Generate base features
    age = np.random.normal(35, 10, n_samples)
    experience = np.random.normal(5, 3, n_samples)
    education_score = np.random.normal(3.0, 1.0, n_samples)

    # Protected attributes
    gender = np.random.choice(["M", "F"], n_samples)
    race = np.random.choice(["White", "Black", "Hispanic", "Asian", "Other"], n_samples)

    # Generate target with optional bias
    base_score = (
        age * 0.02
        + experience * 0.3
        + education_score * 0.4
        + np.random.normal(0, 0.2, n_samples)
    )

    if add_bias:
        # Inject gender bias
        gender_bias = np.where(gender == "M", 0.3, -0.3)
        # Inject racial bias
        race_bias = np.where(race == "White", 0.2, np.where(race == "Asian", 0.1, -0.2))
        base_score += gender_bias + race_bias

    # Convert to binary hiring decision
    hired = (base_score > np.median(base_score)).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "experience": experience,
            "education_score": education_score,
            "gender": gender,
            "race": race,
            "salary_expectation": np.random.normal(60000, 15000, n_samples),
            "interview_score": np.random.normal(3.5, 1.0, n_samples),
            "hired": hired,
        }
    )


def demonstrate_best_practices():
    """
    Demonstrate the complete best practices workflow for CIAF.
    """
    print("🎯 CIAF BEST PRACTICES: Complete Workflow")
    print("=" * 60)

    # Step 1: Generate dataset
    print("\n📊 Step 1: Generate Sample Dataset")
    dataset = create_sample_dataset(n_samples=1000, add_bias=True)
    print(f"✅ Generated dataset with {len(dataset)} samples")
    print(f"📋 Features: {list(dataset.columns)}")
    print(f"📊 Target distribution:\n{dataset['hired'].value_counts(normalize=True)}")

    # Step 2: PRE-INGESTION VALIDATION (Critical Step!)
    print("\n🔍 Step 2: Pre-Ingestion Validation (BEFORE train/test split)")
    validator = PreIngestionValidator()

    # Define protected attributes for bias detection
    protected_attributes = ["gender", "race"]
    target_column = "hired"

    # Run comprehensive validation on COMPLETE dataset
    validation_result = validator.validate_dataset(
        data=dataset,
        target_column=target_column,
        protected_attributes=protected_attributes,
    )

    print(f"⭐ Quality Score: {validation_result['data_quality_score']}/100")
    print(f"🚨 Issues Found: {validation_result['validation_issues']['total_issues']}")
    print(
        f"⚖️ Bias Detected: {validation_result['bias_analysis']['attributes_analyzed']} attributes"
    )
    print(
        f"🚀 Ready for Training: {'✅ Yes' if validation_result['ready_for_training'] else '❌ No'}"
    )

    # Show bias details
    if validation_result["bias_analysis"]["bias_detected"]:
        print("\n🚨 BIAS ANALYSIS:")
        for result in validation_result["bias_analysis"]["results"]:
            if result["bias_detected"]:
                print(
                    f"  • {result['protected_attribute']}: {result['bias_score']:.3f} "
                    f"({'🔴 Critical' if result['bias_score'] > 0.3 else '🟡 Moderate' if result['bias_score'] > 0.1 else '🟢 Low'})"
                )

    # Step 3: Address Issues (if not ready for training)
    if not validation_result["ready_for_training"]:
        print("\n⚠️ Step 3: Dataset Issues Detected")
        print("🔧 In production, you would:")
        print("  • Rebalance the dataset")
        print("  • Collect more representative data")
        print("  • Apply debiasing techniques")
        print("  • Fix data quality issues")
        print("💡 For demo purposes, proceeding with awareness of issues...")

    # Step 4: Train/Test Split (After validation!)
    print("\n✂️ Step 4: Train/Test Split (After Validation)")
    features = dataset.drop(["hired"], axis=1)
    target = dataset["hired"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    print(
        f"📊 Training set: {len(X_train)} samples ({len(X_train)/len(dataset)*100:.1f}%)"
    )
    print(
        f"📊 Testing set: {len(X_test)} samples ({len(X_test)/len(dataset)*100:.1f}%)"
    )

    # Step 5: Initialize Dataset Anchor with Phase Tracking
    print("\n📦 Step 5: Initialize Dataset Anchor with Phase Tracking")
    anchor = DatasetAnchor(
        dataset_id="best_practices_demo", model_name="hiring_model_best_practices"
    )

    # Add all original data to establish full dataset baseline
    print("📥 Adding complete dataset to anchor...")
    for idx, row in dataset.iterrows():
        anchor.add_data_item(
            item_id=f"sample_{idx}",
            content=row.to_dict(),
            metadata={"sample_id": idx, "phase": "full_dataset"},
        )

    # Set phase totals to track split
    anchor.set_phase_totals(training_total=len(X_train), testing_total=len(X_test))

    print(f"✅ Full dataset capsulated: {len(dataset)} samples added")

    # Step 6: Training Phase
    print("\n🏋️ Step 6: Training Phase")

    # Prepare training data (convert categorical to numeric for sklearn)
    X_train_numeric = pd.get_dummies(X_train, drop_first=True)
    X_test_numeric = pd.get_dummies(X_test, drop_first=True)

    # Align columns between train and test
    missing_cols = set(X_train_numeric.columns) - set(X_test_numeric.columns)
    for col in missing_cols:
        X_test_numeric[col] = 0
    X_test_numeric = X_test_numeric[X_train_numeric.columns]

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_numeric, y_train)

    # Simulate capsulation during training (only training samples)
    training_indices = X_train.index
    for idx in training_indices:
        anchor.add_data_item(
            item_id=f"train_sample_{idx}",
            content=X_train.loc[idx].to_dict(),
            metadata={"sample_id": idx, "phase": "training"},
            phase="training",
        )

    print(f"✅ Training completed with {len(X_train)} samples")

    # Step 7: Testing Phase
    print("\n🧪 Step 7: Testing Phase")

    # Make predictions
    y_pred = model.predict(X_test_numeric)
    accuracy = accuracy_score(y_test, y_pred)

    # Simulate capsulation during testing (only testing samples)
    testing_indices = X_test.index
    capsulated_test_samples = np.random.choice(
        testing_indices,
        size=int(len(testing_indices) * 0.8),  # Simulate 80% capsulation
        replace=False,
    )

    for idx in capsulated_test_samples:
        anchor.add_data_item(
            item_id=f"test_sample_{idx}",
            content=X_test.loc[idx].to_dict(),
            metadata={
                "sample_id": idx,
                "phase": "testing",
                "prediction": int(y_pred[X_test.index.get_loc(idx)]),
            },
            phase="testing",
        )

    print(f"✅ Testing completed - Accuracy: {accuracy:.3f}")
    print(f"📊 Capsulated {len(capsulated_test_samples)}/{len(X_test)} test samples")

    # Step 8: Capsulation Analysis
    print("\n📊 Step 8: Capsulation Analysis")
    capsulation_summary = anchor.get_capsulation_summary()

    print("📋 CAPSULATION SUMMARY:")
    print(f"  Total Items Tracked: {capsulation_summary['total_items_tracked']}")
    print(f"  Merkle Tree Samples: {capsulation_summary['merkle_tree_samples']}")

    if "capsulation_status" in capsulation_summary:
        status = capsulation_summary["capsulation_status"]
        print("  Phase-specific tracking:")
        for phase, data in status.items():
            if isinstance(data, dict) and "total" in data:
                print(
                    f"    {phase}: {data.get('capsulated', 0)}/{data['total']} samples"
                )

    print(f"\n💡 {capsulation_summary.get('split_impact_note', 'N/A')}")
    print(f"💡 {capsulation_summary.get('recommendation', 'N/A')}")

    # Step 9: Metadata Integration
    print("\n📝 Step 9: Metadata Integration")
    metadata_manager = create_model_manager("hiring_model_best_practices")

    # Log comprehensive metadata
    metadata_id = metadata_manager.log_event(
        stage="model_development",
        event_type="best_practices_demo",
        metadata={
            "pre_validation": {
                "quality_score": validation_result["data_quality_score"],
                "issues_count": validation_result["validation_issues"]["total_issues"],
                "bias_detected": validation_result["bias_analysis"][
                    "attributes_analyzed"
                ],
                "ready_for_training": validation_result["ready_for_training"],
            },
            "dataset_info": {
                "total_samples": len(dataset),
                "training_samples": len(X_train),
                "testing_samples": len(X_test),
                "features": list(features.columns),
                "target_column": target_column,
                "protected_attributes": protected_attributes,
            },
            "model_performance": {
                "accuracy": accuracy,
                "training_samples_capsulated": len(training_indices),
                "testing_samples_capsulated": len(capsulated_test_samples),
            },
            "capsulation_summary": capsulation_summary,
            "compliance_notes": "Pre-ingestion validation completed before train/test split",
        },
        details="Comprehensive best practices demonstration with pre-ingestion validation",
    )

    print(f"✅ Metadata logged with ID: {metadata_id[:8]}...")

    print("\n🎯 BEST PRACTICES SUMMARY:")
    print("=" * 60)
    print("✅ 1. Run pre-ingestion validation on COMPLETE dataset")
    print("✅ 2. Address bias and quality issues BEFORE splitting")
    print("✅ 3. Use stratified splitting for balanced representation")
    print("✅ 4. Track capsulation by phase, not just overall")
    print("✅ 5. Document validation decisions and split impact")
    print("✅ 6. Integrate comprehensive metadata logging")
    print("\n💡 Remember: <100% capsulation in training/testing phases is")
    print("   EXPECTED due to random splits - validate the complete dataset!")


if __name__ == "__main__":
    demonstrate_best_practices()
