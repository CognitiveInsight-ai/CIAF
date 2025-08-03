"""
CIAF Framework Comprehensive Demo

This demo showcases the reorganized Cognitive Insight AI Framework (CIAF)
with its modular structure and lazy capsule materialization capabilities.
"""

import time
from typing import List, Dict, Any

# Import from reorganized CIAF modules
from ciaf import (
    CIAFFramework, 
    CIAFModelWrapper,
    DatasetAnchor,
    LazyManager,
    ProvenanceCapsule,
    TrainingSnapshot,
    ModelAggregationKey,
    InferenceReceipt,
    ZKEChain,
    CryptoUtils,
    KeyManager,
    MLFrameworkSimulator
)


def generate_synthetic_dataset(num_items: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic training data for demonstration."""
    print(f"🎲 Generating {num_items} synthetic data items...")
    
    # Simulate job posting classification data
    job_categories = ["engineering", "marketing", "sales", "finance", "operations"]
    companies = ["TechCorp", "DataCo", "AIStartup", "BigTech", "Innovation Inc"]
    
    data_items = []
    for i in range(num_items):
        category = job_categories[i % len(job_categories)]
        company = companies[i % len(companies)]
        
        content = f"Job posting #{i+1}: {category.capitalize()} position at {company}"
        
        data_item = {
            "content": content,
            "metadata": {
                "id": f"job_{i+1:03d}",
                "category": category,
                "company": company,
                "posted_date": f"2024-01-{(i % 30) + 1:02d}",
                "target": job_categories.index(category)  # For ML training
            }
        }
        data_items.append(data_item)
    
    print(f"✅ Generated {len(data_items)} synthetic job postings")
    return data_items


def demo_core_crypto_functionality():
    """Demonstrate the core cryptographic functionality."""
    print("\n" + "="*60)
    print("🔐 DEMO: Core Cryptographic Functionality")
    print("="*60)
    
    # Test CryptoUtils
    crypto = CryptoUtils()
    test_data = b"Hello, CIAF World!"
    
    # Hash functionality
    hash_result = crypto.sha256_hash(test_data)
    print(f"SHA256 Hash: {hash_result}")
    
    # Key derivation
    key_manager = KeyManager()
    master_key = key_manager.derive_key_pbkdf2("demo_password", b"demo_salt", 32)
    print(f"Derived Key: {master_key.hex()[:32]}...")
    
    # HMAC functionality
    hmac_result = crypto.hmac_sha256(master_key, test_data)
    print(f"HMAC-SHA256: {hmac_result}")
    
    print("✅ Core cryptographic tests completed")


def demo_dataset_anchoring_and_lazy_management():
    """Demonstrate dataset anchoring with lazy management."""
    print("\n" + "="*60)
    print("⚓ DEMO: Dataset Anchoring and Lazy Management")
    print("="*60)
    
    # Create synthetic dataset
    training_data = generate_synthetic_dataset(500)
    
    # Create dataset anchor
    dataset_metadata = {
        "name": "Job Classification Dataset",
        "version": "1.0",
        "description": "Synthetic job posting classification data"
    }
    
    anchor = DatasetAnchor(
        dataset_id="job_dataset_v1",
        metadata=dataset_metadata,
        master_password="demo_master_password",
        salt=b"job_dataset_salt"
    )
    
    # Add data items to anchor
    for item in training_data:
        anchor.add_data_item(
            item_id=item['metadata']['id'],
            content=item['content'],
            metadata=item['metadata']
        )
    
    print(f"📊 Dataset anchor created with {len(anchor.data_items)} items")
    print(f"🔑 Dataset key derived: {anchor.dataset_key[:32]}...")
    
    # Demonstrate lazy management
    lazy_manager = LazyManager(anchor)
    
    # Create some lazy capsules
    start_time = time.time()
    capsules = []
    for i, item in enumerate(training_data[:10]):  # Create 10 lazy capsules
        capsule = lazy_manager.create_lazy_capsule(
            item_id=item['metadata']['id'],
            original_data=item['content'],
            metadata=item['metadata']
        )
        capsules.append(capsule)
    
    lazy_time = time.time() - start_time
    print(f"⚡ Created {len(capsules)} lazy capsules in {lazy_time:.4f} seconds")
    print(f"📈 Materialized capsules: {len(lazy_manager.materialized_capsules)}")
    
    return anchor, lazy_manager, capsules


def demo_ml_framework_integration():
    """Demonstrate ML framework integration with CIAF."""
    print("\n" + "="*60)
    print("🤖 DEMO: ML Framework Integration")
    print("="*60)
    
    # Create ML framework simulator
    ml_simulator = MLFrameworkSimulator("JobClassifier")
    
    # Generate training data
    training_data = generate_synthetic_dataset(1000)

    # Create mock data secrets (in real scenario, these would be secure)
    data_secrets = {item['metadata']['id']: f"secret_{item['metadata']['id']}" 
                   for item in training_data}
    
    # Prepare data for training (creates provenance capsules)
    capsules = ml_simulator.prepare_data_for_training(training_data, data_secrets)
    print(f"📦 Created {len(capsules)} provenance capsules")
    
    # Create Model Aggregation Key
    mak = ModelAggregationKey(
        key_id="JobClassifier_MAK",
        secret_material="demo_secret_material_for_job_classifier"
    )
    
    # Train model
    training_params = {
        "algorithm": "RandomForest",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    
    snapshot = ml_simulator.train_model(
        training_data_capsules=capsules,
        mak=mak,
        training_params=training_params,
        model_version="1.0.0"
    )
    
    print(f"🎯 Training snapshot created: {snapshot.snapshot_id}")
    print(f"📊 Snapshot covers {len(snapshot.provenance_capsule_hashes)} data items")
    
    return ml_simulator, snapshot, mak


def demo_complete_ciaf_workflow():
    """Demonstrate the complete CIAF framework workflow."""
    print("\n" + "="*60)
    print("🚀 DEMO: Complete CIAF Framework Workflow")
    print("="*60)
    
    # Initialize CIAF Framework
    framework = CIAFFramework("CIAF_Demo")
    
    # Generate training data
    training_data = generate_synthetic_dataset(1500)
    
    # Create dataset anchor
    dataset_id = "complete_job_dataset"
    dataset_metadata = {
        "name": "Complete Job Classification Dataset",
        "version": "2.0",
        "demo": True
    }
    
    anchor = framework.create_dataset_anchor(
        dataset_id=dataset_id,
        dataset_metadata=dataset_metadata,
        master_password="ciaf_demo_password"
    )
    
    # Create provenance capsules with lazy materialization
    capsules = framework.create_provenance_capsules(dataset_id, training_data)
    
    # Create Model Aggregation Key
    mak = framework.create_model_aggregation_key(
        model_name="CIAFJobClassifier",
        authorized_datasets=[dataset_id]
    )
    
    # Train model
    training_params = {
        "model_type": "deep_learning",
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001
    }
    
    snapshot = framework.train_model(
        model_name="CIAFJobClassifier",
        capsules=capsules,
        mak=mak,
        training_params=training_params,
        model_version="2.0.0"
    )
    
    # Validate training integrity
    is_valid = framework.validate_training_integrity(snapshot)
    print(f"🔍 Training integrity validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    
    # Get performance metrics
    metrics = framework.get_performance_metrics(dataset_id)
    print(f"📈 Performance metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value}")
    
    return framework, snapshot


def demo_inference_receipts_and_chaining():
    """Demonstrate inference receipts and chaining."""
    print("\n" + "="*60)
    print("🔗 DEMO: Inference Receipts and Chaining")
    print("="*60)
    
    # Create a ZKE Chain for receipt management
    chain = ZKEChain()
    
    # Mock training snapshot data
    training_snapshot_id = "demo_snapshot_123"
    merkle_root = "mock_merkle_root_hash_456"
    model_version = "1.0.0"
    
    # Generate multiple inference receipts
    queries = [
        "What are the requirements for a software engineer position?",
        "How much does a marketing manager earn?",
        "What skills are needed for a data scientist role?",
        "Are remote positions available?",
        "What benefits are offered?"
    ]
    
    print(f"🔮 Creating inference receipts for {len(queries)} queries...")
    
    for i, query in enumerate(queries):
        ai_output = f"Simulated AI response #{i+1} for: {query}"
        
        receipt = chain.add_receipt(
            query=query,
            ai_output=ai_output,
            model_version=model_version,
            training_snapshot_id=training_snapshot_id,
            training_snapshot_merkle_root=merkle_root
        )
        
        print(f"📋 Receipt {i+1}: {receipt.receipt_hash[:16]}...")
    
    # Verify chain integrity
    chain_valid = chain.verify_chain()
    print(f"🔍 Chain integrity verification: {'✅ VALID' if chain_valid else '❌ INVALID'}")
    
    # Get chain summary
    summary = chain.get_chain_summary()
    print(f"📊 Chain summary:")
    for key, value in summary.items():
        print(f"    {key}: {value}")
    
    return chain


def demo_model_wrapper():
    """Demonstrate the CIAF model wrapper for drop-in integration."""
    print("\n" + "="*60)
    print("🎁 DEMO: CIAF Model Wrapper (Drop-in Integration)")
    print("="*60)
    
    # Simulate a scikit-learn-like model
    class MockMLModel:
        def __init__(self):
            self.is_fitted = False
        
        def fit(self, X, y):
            print(f"    MockMLModel.fit() called with {len(X)} samples")
            self.is_fitted = True
            return self
        
        def predict(self, X):
            if not self.is_fitted:
                raise ValueError("Model not fitted")
            return [f"prediction_for_{x}" for x in X]
    
    # Create wrapped model
    mock_model = MockMLModel()
    wrapper = CIAFModelWrapper(
        model=mock_model,
        model_name="WrappedJobClassifier",
        enable_chaining=True,
        compliance_mode="general"
    )
    
    # Generate training data
    training_data = generate_synthetic_dataset(2000)
    
    # Train with CIAF wrapper
    snapshot = wrapper.train(
        dataset_id="wrapped_model_dataset",
        training_data=training_data,
        master_password="wrapper_demo_password",
        training_params={"mock_param": "mock_value"},
        model_version="1.0.0",
        fit_model=True
    )
    
    print(f"🎯 Wrapped model training completed: {snapshot.snapshot_id}")
    
    # Make predictions with automatic receipt generation
    test_queries = [
        "Engineering job requirements",
        "Marketing salary range"
    ]
    
    for query in test_queries:
        prediction, receipt = wrapper.predict(query, use_model=False)  # Use simulator
        print(f"🔮 Query: {query}")
        print(f"    Prediction: {prediction}")
        print(f"    Receipt: {receipt.receipt_hash[:16]}...")
    
    # Get model info
    model_info = wrapper.get_model_info()
    print(f"📊 Model info:")
    for key, value in model_info.items():
        print(f"    {key}: {value}")
    
    return wrapper


def run_comprehensive_demo():
    """Run the complete CIAF demonstration."""
    print("🎉 CIAF Framework Comprehensive Demo")
    print("=" * 60)
    print("This demo showcases the reorganized CIAF framework with modular architecture")
    print("and lazy capsule materialization for improved performance.")
    print()
    
    start_time = time.time()
    
    try:
        # Run all demo sections
        demo_core_crypto_functionality()
        anchor, lazy_manager, capsules = demo_dataset_anchoring_and_lazy_management()
        ml_simulator, snapshot, mak = demo_ml_framework_integration()
        framework, complete_snapshot = demo_complete_ciaf_workflow()
        chain = demo_inference_receipts_and_chaining()
        wrapper = demo_model_wrapper()
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("🎊 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print()
        print("✅ Demonstrated features:")
        print("   • Modular CIAF architecture")
        print("   • Core cryptographic operations")
        print("   • Dataset anchoring with lazy management")
        print("   • ML framework integration")
        print("   • Complete workflow orchestration")
        print("   • Inference receipt chaining")
        print("   • Drop-in model wrapper")
        print()
        print("🚀 The CIAF framework is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    run_comprehensive_demo()
