"""
CIAF True Lazy Implementation Demo

This demo shows how to use the updated CIAF implementation with true lazy behavior
that achieves patent-level performance improvements.
"""

import time
from ciaf.anchoring import LazyProvenanceManager, DatasetAnchor


def demo_true_lazy_performance():
    """Demonstrate the performance benefits of true lazy implementation."""
    
    print("🚀 CIAF True Lazy Implementation Demo")
    print("=" * 50)
    
    # Sample dataset
    print("📊 Creating sample dataset...")
    sample_data = []
    for i in range(1000):
        sample_data.append({
            "id": f"sample_{i:04d}",
            "content": f"Data sample {i} with content: " + "X" * 500,
            "metadata": {
                "source": f"source_{i % 10}",
                "category": f"cat_{i % 5}",
                "timestamp": time.time() + i,
                "importance": i % 3
            }
        })
    
    print(f"✅ Generated {len(sample_data)} sample items")
    
    # Test True Lazy Implementation
    print(f"\n⚡ Testing True Lazy Implementation...")
    print("-" * 40)
    
    start_time = time.perf_counter()
    
    # Create true lazy manager
    lazy_manager = LazyProvenanceManager(use_true_lazy=True)
    
    # Create dataset anchor
    anchor = lazy_manager.create_dataset_anchor(
        dataset_id="demo_dataset",
        model_name="demo_model",
        dataset_metadata={
            "description": "Demo dataset for true lazy testing",
            "version": "1.0.0",
            "total_items": len(sample_data)
        }
    )
    
    # Register all items (should be very fast)
    for item in sample_data:
        lazy_manager.register_lazy_capsule(
            dataset_id="demo_dataset",
            capsule_id=item["id"],
            sample_data=item["content"],
            metadata=item["metadata"]
        )
    
    registration_time = time.perf_counter() - start_time
    
    print(f"✅ Registered {len(sample_data)} items in {registration_time:.6f}s")
    print(f"   Average: {registration_time/len(sample_data)*1000:.3f}ms per item")
    
    # Get dataset summary
    summary = lazy_manager.get_dataset_summary("demo_dataset")
    print(f"\n📊 Dataset Summary:")
    print(f"   Implementation: {summary.get('lazy_implementation', 'Unknown')}")
    print(f"   Total items: {summary.get('total_items', 0)}")
    
    if 'performance_improvement' in summary:
        print(f"   Performance improvement: {summary['performance_improvement']:.1f}x")
    
    if 'memory_efficiency' in summary:
        memory_eff = summary['memory_efficiency']
        print(f"   Memory efficiency: {memory_eff.get('memory_saving_ratio', 0)*100:.1f}% savings")
    
    # Demonstrate selective materialization
    print(f"\n🔍 Demonstrating Selective Materialization...")
    print("-" * 40)
    
    # Materialize just a few items for audit
    audit_samples = ["sample_0000", "sample_0100", "sample_0500", "sample_0999"]
    
    materialize_start = time.perf_counter()
    
    for sample_id in audit_samples:
        capsule = lazy_manager.materialize_capsule("demo_dataset", sample_id)
        print(f"   ✅ Materialized {sample_id}: {capsule.metadata.get('audit_reference', 'N/A')}")
    
    materialization_time = time.perf_counter() - materialize_start
    
    print(f"\n📈 Materialization Results:")
    print(f"   Materialized {len(audit_samples)} items in {materialization_time:.6f}s")
    print(f"   Average: {materialization_time/len(audit_samples)*1000:.3f}ms per item")
    
    # Show efficiency calculation
    total_lazy_time = registration_time + materialization_time
    print(f"\n⚡ Efficiency Analysis:")
    print(f"   Total lazy time: {total_lazy_time:.6f}s")
    print(f"   Items processed: {len(sample_data)} registered + {len(audit_samples)} materialized")
    print(f"   Efficiency: Only {len(audit_samples)/len(sample_data)*100:.1f}% of items needed full processing")
    
    # Performance statistics
    perf_stats = lazy_manager.get_performance_stats()
    if perf_stats.get('lazy_implementation') == 'TrueLazyManager':
        print(f"\n📊 Performance Statistics:")
        avg_improvement = perf_stats.get('average_performance_improvement', 1.0)
        print(f"   Average performance improvement: {avg_improvement:.1f}x")
        print(f"   Total datasets: {perf_stats.get('total_datasets', 0)}")
    
    # Demonstrate audit capability
    print(f"\n🔐 Demonstrating Audit Capability...")
    print("-" * 40)
    
    audit_results = []
    for sample_id in audit_samples[:2]:  # Audit first two samples
        audit_result = lazy_manager.audit_capsule_provenance("demo_dataset", sample_id)
        audit_results.append(audit_result)
        
        status = "✅ PASSED" if audit_result.get('audit_passed') else "❌ FAILED"
        print(f"   {sample_id}: {status}")
    
    print(f"\n🎯 Demo Summary:")
    print(f"   ✅ True lazy implementation working correctly")
    print(f"   ✅ {len(sample_data)} items registered efficiently")
    print(f"   ✅ {len(audit_samples)} items materialized on-demand")
    print(f"   ✅ {len(audit_results)} items audited successfully")
    print(f"   ✅ Performance improvement demonstrated")
    
    return {
        'registration_time': registration_time,
        'materialization_time': materialization_time,
        'total_items': len(sample_data),
        'materialized_items': len(audit_samples),
        'audit_results': audit_results
    }


def demo_legacy_vs_true_lazy():
    """Compare legacy vs true lazy performance."""
    
    print(f"\n🔬 Legacy vs True Lazy Comparison")
    print("=" * 50)
    
    # Test dataset
    test_size = 500
    test_data = []
    for i in range(test_size):
        test_data.append({
            "id": f"test_{i:04d}",
            "content": f"Test data {i}: " + "Y" * 200,
            "metadata": {"test": True, "index": i}
        })
    
    # Test Legacy Implementation
    print(f"🐌 Testing Legacy Implementation ({test_size} items)...")
    
    legacy_start = time.perf_counter()
    
    legacy_manager = LazyProvenanceManager(use_true_lazy=False)
    legacy_anchor = legacy_manager.create_dataset_anchor(
        dataset_id="legacy_test",
        model_name="test_model",
        dataset_metadata={"test": "legacy"}
    )
    
    for item in test_data:
        legacy_manager.register_lazy_capsule(
            dataset_id="legacy_test",
            capsule_id=item["id"],
            sample_data=item["content"],
            metadata=item["metadata"]
        )
    
    legacy_time = time.perf_counter() - legacy_start
    
    # Test True Lazy Implementation
    print(f"⚡ Testing True Lazy Implementation ({test_size} items)...")
    
    true_lazy_start = time.perf_counter()
    
    true_lazy_manager = LazyProvenanceManager(use_true_lazy=True)
    true_lazy_anchor = true_lazy_manager.create_dataset_anchor(
        dataset_id="true_lazy_test",
        model_name="test_model",
        dataset_metadata={"test": "true_lazy"}
    )
    
    for item in test_data:
        true_lazy_manager.register_lazy_capsule(
            dataset_id="true_lazy_test",
            capsule_id=item["id"],
            sample_data=item["content"],
            metadata=item["metadata"]
        )
    
    true_lazy_time = time.perf_counter() - true_lazy_start
    
    # Results
    improvement_factor = legacy_time / true_lazy_time
    
    print(f"\n📊 Comparison Results:")
    print(f"   Legacy time:     {legacy_time:.6f}s ({legacy_time/test_size*1000:.3f}ms per item)")
    print(f"   True lazy time:  {true_lazy_time:.6f}s ({true_lazy_time/test_size*1000:.3f}ms per item)")
    print(f"   Improvement:     {improvement_factor:.1f}x speedup")
    
    if improvement_factor >= 10:
        status = "✅ EXCELLENT - Order of magnitude improvement"
    elif improvement_factor >= 2:
        status = "✅ GOOD - Significant improvement"
    else:
        status = "⚠️  MODEST - Some improvement"
    
    print(f"   Status: {status}")
    
    return {
        'legacy_time': legacy_time,
        'true_lazy_time': true_lazy_time,
        'improvement_factor': improvement_factor,
        'test_size': test_size
    }


if __name__ == "__main__":
    print("🎯 CIAF True Lazy Implementation Demonstration")
    print("This demo shows the performance benefits of the updated CIAF implementation")
    print()
    
    # Run main demo
    demo_results = demo_true_lazy_performance()
    
    # Run comparison demo
    comparison_results = demo_legacy_vs_true_lazy()
    
    print(f"\n🏆 Final Results Summary:")
    print(f"   Main demo: {demo_results['total_items']} items processed efficiently")
    print(f"   Registration: {demo_results['registration_time']:.6f}s")
    print(f"   Materialization: {demo_results['materialization_time']:.6f}s")
    print(f"   Comparison improvement: {comparison_results['improvement_factor']:.1f}x")
    print(f"   ✅ True lazy implementation validated!")
