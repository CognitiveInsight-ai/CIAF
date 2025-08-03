#!/usr/bin/env python3
"""
Pre-Ingestion Validation and Data Capsulation Demo

This script demonstrates the correct approach to handle bias detection and 
data validation BEFORE train/test split, and properly track capsulation 
percentages that account for the random split.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

# Add CIAF to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from ciaf.compliance.pre_ingestion_validator import validate_before_split
    from ciaf.anchoring.dataset_anchor import DatasetAnchor
    from ciaf.metadata_integration import ModelMetadataManager
    CIAF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CIAF import error: {e}")
    CIAF_AVAILABLE = False


def generate_biased_dataset(n_samples=1000):
    """Generate a dataset with intentional bias for demonstration."""
    print(f"📊 Generating biased dataset with {n_samples} samples...")
    
    np.random.seed(42)
    
    # Protected attributes
    gender = np.random.choice(['Male', 'Female', 'Non-binary'], n_samples, p=[0.6, 0.35, 0.05])
    race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], n_samples, 
                           p=[0.6, 0.15, 0.15, 0.08, 0.02])
    age = np.random.normal(35, 10, n_samples)
    
    # Features
    education_score = np.random.normal(7.5, 1.5, n_samples)
    experience_years = np.random.exponential(3, n_samples)
    skill_assessment = np.random.normal(8.0, 2.0, n_samples)
    
    # Introduce intentional bias in hiring decisions
    bias_multiplier = np.where(gender == 'Male', 1.2,  # 20% bias toward males
                              np.where(gender == 'Female', 0.9, 1.0))  # 10% bias against females
    
    race_bias = np.where(race == 'White', 1.1,  # 10% bias toward white candidates
                        np.where(np.isin(race, ['Black', 'Hispanic']), 0.85, 1.0))  # 15% bias against minorities
    
    # Calculate hiring probability with bias
    base_score = (0.3 * education_score/10 + 0.4 * np.minimum(experience_years/10, 1.0) + 0.3 * skill_assessment/10)
    biased_score = base_score * bias_multiplier * race_bias
    
    # Add some noise and create binary outcome
    biased_score += np.random.normal(0, 0.1, n_samples)
    hired = (biased_score > 0.6).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'education_score': education_score,
        'experience_years': experience_years,
        'skill_assessment': skill_assessment,
        'age': age,
        'gender': gender,
        'race': race,
        'hired': hired,
        'candidate_id': [f"candidate_{i:04d}" for i in range(n_samples)]
    })
    
    # Add some missing values and duplicates for quality testing
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    data.loc[missing_indices, 'education_score'] = np.nan
    
    # Add a few duplicate rows
    duplicate_indices = np.random.choice(n_samples-10, size=5, replace=False)
    for idx in duplicate_indices:
        data.loc[n_samples + len(duplicate_indices)] = data.loc[idx]
    
    print(f"✅ Generated dataset with intentional bias and quality issues")
    return data


def demonstrate_pre_ingestion_validation():
    """Demonstrate pre-ingestion validation before any train/test split."""
    print("\n" + "="*80)
    print("🔍 PRE-INGESTION VALIDATION DEMONSTRATION")
    print("="*80)
    
    # Generate biased dataset
    data = generate_biased_dataset(1000)
    
    print(f"\n📊 Original dataset shape: {data.shape}")
    print(f"📋 Target distribution:\n{data['hired'].value_counts(normalize=True)}")
    
    if not CIAF_AVAILABLE:
        print("❌ CIAF not available - skipping validation")
        return data
    
    # Perform comprehensive pre-ingestion validation
    print(f"\n🔍 Running pre-ingestion validation...")
    validation_report = validate_before_split(
        data=data,
        target_column='hired',
        protected_attributes=['gender', 'race'],
        compliance_framework='EEOC',
        sensitive_columns=['candidate_id', 'age']
    )
    
    # Show key findings
    if validation_report['bias_analysis']['bias_detected']:
        print("\n🚨 BIAS DETECTED - Action required before proceeding!")
        for result in validation_report['bias_analysis']['results']:
            if result['bias_detected']:
                print(f"  • {result['protected_attribute']}: {result['bias_score']:.3f} bias score")
                print(f"    Recommendation: {result['recommendation']}")
    
    if not validation_report['ready_for_training']:
        print("\n❌ Dataset NOT ready for training - address issues first!")
        critical_issues = [
            issue for issue in validation_report['validation_issues']['details']
            if issue['severity'] == 'critical'
        ]
        for issue in critical_issues:
            print(f"  🔴 {issue['message']}")
    else:
        print(f"\n✅ Dataset ready for training (Quality Score: {validation_report['data_quality_score']}/100)")
    
    return data, validation_report


def demonstrate_capsulation_tracking():
    """Demonstrate proper capsulation tracking that accounts for train/test split."""
    print("\n" + "="*80)
    print("📦 DATA CAPSULATION TRACKING DEMONSTRATION")
    print("="*80)
    
    if not CIAF_AVAILABLE:
        print("❌ CIAF not available - skipping capsulation demo")
        return
    
    # Generate clean dataset for capsulation demo
    data = generate_biased_dataset(500)  # Smaller for demo
    
    # Clean the data (in practice, you'd fix issues found in validation)
    data_clean = data.dropna().drop_duplicates().reset_index(drop=True)
    print(f"📊 Cleaned dataset: {len(data_clean)} samples")
    
    # Create dataset anchor BEFORE train/test split
    print(f"\n📦 Creating dataset anchor for full dataset...")
    anchor = DatasetAnchor(
        dataset_id="hiring_model_demo",
        metadata={
            "model_type": "hiring_classifier",
            "original_samples": len(data),
            "cleaned_samples": len(data_clean),
            "validation_completed": True
        },
        master_password="demo_password_2024"
    )
    
    # Add all samples to the anchor (full dataset phase)
    print(f"📥 Adding all samples to dataset anchor...")
    for idx, row in data_clean.iterrows():
        anchor.add_data_item(
            item_id=f"sample_{idx}",
            content=f"candidate_data_{row['candidate_id']}",
            metadata={
                "education_score": row['education_score'],
                "experience_years": row['experience_years'],
                "hired": row['hired'],
                "gender": row['gender'],
                "race": row['race']
            },
            phase='full_dataset'
        )
    
    # Now perform train/test split
    print(f"\n✂️ Performing train/test split...")
    X = data_clean.drop(['hired', 'candidate_id'], axis=1)
    y = data_clean['hired']
    
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, data_clean.index, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Split results:")
    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(data_clean)*100:.1f}%)")
    print(f"  Testing set: {len(X_test)} samples ({len(X_test)/len(data_clean)*100:.1f}%)")
    
    # Set the phase totals in the anchor
    anchor.set_phase_totals(len(X_train), len(X_test))
    
    # Simulate capsulation during training phase
    print(f"\n🏋️ Simulating training phase capsulation...")
    for idx in indices_train:
        row = data_clean.loc[idx]
        anchor.add_data_item(
            item_id=f"train_sample_{idx}",
            content=f"training_data_{row['candidate_id']}",
            metadata={
                "phase": "training",
                "education_score": row['education_score'],
                "hired": row['hired']
            },
            phase='training_phase'
        )
    
    # Simulate capsulation during testing phase (partial for demo)
    test_sample_size = int(len(indices_test) * 0.8)  # Simulate 80% of test samples processed
    print(f"\n🧪 Simulating testing phase capsulation ({test_sample_size}/{len(indices_test)} samples)...")
    for idx in list(indices_test)[:test_sample_size]:
        row = data_clean.loc[idx]
        anchor.add_data_item(
            item_id=f"test_sample_{idx}",
            content=f"testing_data_{row['candidate_id']}",
            metadata={
                "phase": "testing",
                "education_score": row['education_score'],
                "hired": row['hired']
            },
            phase='testing_phase'
        )
    
    # Show capsulation summary
    print(f"\n📊 CAPSULATION SUMMARY:")
    summary = anchor.get_capsulation_summary()
    
    for phase, status in summary['capsulation_status'].items():
        if status['total'] > 0:
            print(f"  {phase.replace('_', ' ').title()}:")
            print(f"    Capsulated: {status['capsulated']}/{status['total']} ({status['percentage']:.1f}%)")
    
    print(f"\n💡 {summary['split_impact_note']}")
    print(f"💡 {summary['recommendation']}")
    
    return anchor


def demonstrate_metadata_integration():
    """Demonstrate metadata integration with pre-validation and capsulation tracking."""
    print("\n" + "="*80)
    print("📋 METADATA INTEGRATION DEMONSTRATION")
    print("="*80)
    
    if not CIAF_AVAILABLE:
        print("❌ CIAF not available - skipping metadata integration demo")
        return
    
    # Initialize metadata manager
    metadata_manager = ModelMetadataManager("hiring_model_with_validation", "3.0.0")
    
    # Log pre-ingestion validation
    validation_start = datetime.now()
    data, validation_report = demonstrate_pre_ingestion_validation()
    
    validation_id = metadata_manager.log_event(
        stage="pre_validation",
        event_type="bias_and_quality_validation",
        metadata={
            "validation_framework": "EEOC",
            "quality_score": validation_report['data_quality_score'],
            "bias_detected": validation_report['bias_analysis']['bias_detected'],
            "issues_found": validation_report['validation_issues']['total_issues'],
            "ready_for_training": validation_report['ready_for_training'],
            "validation_duration": (datetime.now() - validation_start).total_seconds()
        },
        details="Comprehensive pre-ingestion validation completed before train/test split"
    )
    
    print(f"\n📝 Logged pre-validation metadata: {validation_id[:8]}...")
    
    # Log train/test split with capsulation tracking
    split_id = metadata_manager.log_event(
        stage="data_preparation",
        event_type="train_test_split_with_capsulation",
        metadata={
            "total_samples": len(data),
            "train_test_ratio": "80/20",
            "stratified": True,
            "random_state": 42,
            "capsulation_note": "Capsulation percentages reflect random split impact",
            "split_method": "sklearn.model_selection.train_test_split"
        },
        details="Random train/test split performed with capsulation tracking"
    )
    
    print(f"📝 Logged train/test split metadata: {split_id[:8]}...")
    
    # Log final recommendation
    recommendation_id = metadata_manager.log_event(
        stage="compliance",
        event_type="split_impact_documentation",
        metadata={
            "issue_identified": "Train/test split affects capsulation percentages",
            "root_cause": "Random sampling means only subset of data used in each phase",
            "solution": "Pre-ingestion validation on complete dataset",
            "implementation": "Use pre_ingestion_validator before any data splitting",
            "compliance_impact": "Ensures bias detection on representative complete dataset"
        },
        details="Documented train/test split impact on capsulation and implemented solution"
    )
    
    print(f"📝 Logged compliance documentation: {recommendation_id[:8]}...")
    
    return metadata_manager


def main():
    """Main demonstration function."""
    print("="*80)
    print("🔍 PRE-INGESTION VALIDATION & CAPSULATION TRACKING DEMO")
    print("="*80)
    print("This demo shows the correct approach to handle:")
    print("• Bias detection BEFORE train/test split")
    print("• Proper capsulation tracking that accounts for random splits")
    print("• Comprehensive metadata integration")
    
    try:
        # 1. Pre-ingestion validation
        data, validation_report = demonstrate_pre_ingestion_validation()
        
        # 2. Capsulation tracking
        anchor = demonstrate_capsulation_tracking()
        
        # 3. Metadata integration
        metadata_manager = demonstrate_metadata_integration()
        
        print("\n" + "="*80)
        print("✅ DEMONSTRATION COMPLETE")
        print("="*80)
        
        print("\n🎯 Key Takeaways:")
        print("• ✅ Always validate complete dataset BEFORE train/test split")
        print("• ✅ Use pre_ingestion_validator to catch bias early")
        print("• ✅ Document that capsulation percentages reflect split impact")
        print("• ✅ Track phase-specific capsulation separately")
        print("• ✅ Implement comprehensive metadata logging")
        
        print("\n💡 Best Practices:")
        print("• Run bias detection on the complete dataset")
        print("• Fix data quality issues before splitting")
        print("• Use stratified splitting for balanced representation")
        print("• Document all validation and split decisions")
        print("• Track capsulation by phase, not just overall")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
