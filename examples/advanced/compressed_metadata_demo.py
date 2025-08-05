#!/usr/bin/env python3
"""
CIAF Compressed Metadata Storage Demo

This demo showcases the optimized compressed metadata storage system,
comparing it with the original JSON storage format.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add CIAF to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from ciaf.metadata_storage import MetadataStorage
    from ciaf.metadata_storage_compressed import CompressedMetadataStorage
except ImportError as e:
    print(f"❌ Could not import CIAF modules: {e}")
    sys.exit(1)


def generate_sample_metadata(size: int = 1000) -> dict:
    """Generate realistic metadata for testing."""
    import random

    return {
        "model_info": {
            "algorithm": "RandomForestClassifier",
            "parameters": {
                "n_estimators": random.randint(50, 200),
                "max_depth": random.randint(5, 20),
                "min_samples_split": random.randint(2, 10),
                "random_state": 42,
            },
            "hyperparameters": {f"param_{i}": random.random() for i in range(20)},
        },
        "training_data": {
            "samples": random.randint(1000, 10000),
            "features": random.randint(20, 100),
            "target_classes": random.randint(2, 10),
            "data_hash": f"sha256_{random.getrandbits(128):032x}",
        },
        "performance_metrics": {
            "accuracy": random.uniform(0.8, 0.99),
            "precision": random.uniform(0.8, 0.99),
            "recall": random.uniform(0.8, 0.99),
            "f1_score": random.uniform(0.8, 0.99),
            "auc_roc": random.uniform(0.8, 0.99),
            "confusion_matrix": [
                [random.randint(10, 100) for _ in range(3)] for _ in range(3)
            ],
        },
        "feature_importance": {f"feature_{i}": random.random() for i in range(50)},
        "training_history": {
            "epochs": [i for i in range(random.randint(10, 50))],
            "loss": [random.uniform(0.1, 1.0) for _ in range(random.randint(10, 50))],
            "accuracy": [
                random.uniform(0.7, 0.99) for _ in range(random.randint(10, 50))
            ],
        },
        "environment": {
            "python_version": "3.12.0",
            "scikit_learn_version": "1.4.0",
            "ciaf_version": "2.1.0",
            "system_info": {"platform": "Windows-10", "cpu_count": 8, "memory_gb": 16},
        },
        "compliance_info": {
            "frameworks": ["EU_AI_ACT", "GDPR", "NIST_AI_RMF"],
            "audit_trail": [f"event_{i}" for i in range(random.randint(5, 20))],
            "documentation_links": [
                f"https://docs.example.com/doc_{i}" for i in range(5)
            ],
        },
        "large_data_array": [
            random.random() for _ in range(size)
        ],  # Variable size data
        "timestamp": datetime.now().isoformat(),
        "description": "A" * random.randint(100, 1000),  # Variable text data
    }


def benchmark_storage_performance():
    """Compare performance between JSON and compressed storage."""
    print("🔬 CIAF Compressed Metadata Storage Performance Demo")
    print("=" * 60)

    # Test parameters
    num_records = 50
    data_sizes = [100, 500, 1000, 2000]  # Different sizes for metadata

    print(f"📊 Testing with {num_records} records across different data sizes")
    print()

    results = {}

    for data_size in data_sizes:
        print(f"🧪 Testing with data size: {data_size} elements")

        # Generate test data
        test_data = [generate_sample_metadata(data_size) for _ in range(num_records)]

        # Test original JSON storage
        print("  📁 Testing JSON storage...")
        json_storage = MetadataStorage(
            storage_path=f"benchmark_json_{data_size}", backend="json"
        )

        start_time = time.time()
        json_ids = []
        for i, data in enumerate(test_data):
            metadata_id = json_storage.save_metadata(
                model_name=f"test_model_{i}",
                stage="benchmark",
                event_type="test",
                metadata=data,
            )
            json_ids.append(metadata_id)
        json_write_time = time.time() - start_time

        # Read performance for JSON
        start_time = time.time()
        for metadata_id in json_ids:
            json_storage.get_metadata(metadata_id)
        json_read_time = time.time() - start_time

        # Calculate JSON storage size
        json_size = sum(
            f.stat().st_size
            for f in Path(f"benchmark_json_{data_size}").rglob("*.json")
        ) / (
            1024 * 1024
        )  # MB

        # Test compressed storage with different algorithms
        compression_results = {}

        for compression in ["gzip", "lzma", "zlib"]:
            print(f"  🗜️ Testing {compression} compression...")

            compressed_storage = CompressedMetadataStorage(
                storage_path=f"benchmark_compressed_{compression}_{data_size}",
                compression=compression,
                serialization="json",
            )

            start_time = time.time()
            compressed_ids = []
            for i, data in enumerate(test_data):
                metadata_id = compressed_storage.save_metadata(
                    model_name=f"test_model_{i}",
                    stage="benchmark",
                    event_type="test",
                    metadata=data,
                )
                compressed_ids.append(metadata_id)
            compressed_write_time = time.time() - start_time

            # Read performance for compressed
            start_time = time.time()
            for metadata_id in compressed_ids:
                compressed_storage.get_metadata(metadata_id)
            compressed_read_time = time.time() - start_time

            # Get compression stats
            stats = compressed_storage.get_compression_stats()
            compressed_size = stats["total_compressed_size"] / (1024 * 1024)  # MB

            compression_results[compression] = {
                "write_time": compressed_write_time,
                "read_time": compressed_read_time,
                "compressed_size_mb": compressed_size,
                "compression_ratio": stats["compression_ratio"],
                "space_saved_mb": stats["space_saved_mb"],
            }

        results[data_size] = {
            "json": {
                "write_time": json_write_time,
                "read_time": json_read_time,
                "size_mb": json_size,
            },
            "compressed": compression_results,
        }

        print(
            f"    ✅ JSON: Write {json_write_time:.3f}s, Read {json_read_time:.3f}s, Size {json_size:.2f}MB"
        )
        for comp, result in compression_results.items():
            print(
                f"    ✅ {comp.upper()}: Write {result['write_time']:.3f}s, "
                f"Read {result['read_time']:.3f}s, Size {result['compressed_size_mb']:.2f}MB "
                f"({result['compression_ratio']*100:.1f}% compression)"
            )
        print()

    # Generate summary report
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)

    for data_size, result in results.items():
        print(f"\n📈 Data Size: {data_size} elements per record")
        print("-" * 40)

        json_result = result["json"]
        print(f"JSON Storage:")
        print(
            f"  • Write: {json_result['write_time']:.3f}s ({json_result['write_time']/num_records*1000:.1f}ms/record)"
        )
        print(
            f"  • Read:  {json_result['read_time']:.3f}s ({json_result['read_time']/num_records*1000:.1f}ms/record)"
        )
        print(f"  • Size:  {json_result['size_mb']:.2f}MB")

        best_compression = min(
            result["compressed"].items(),
            key=lambda x: x[1]["write_time"] + x[1]["read_time"],
        )

        best_space = max(
            result["compressed"].items(), key=lambda x: x[1]["compression_ratio"]
        )

        print(f"\nBest Performance ({best_compression[0].upper()}):")
        best_result = best_compression[1]
        print(
            f"  • Write: {best_result['write_time']:.3f}s ({best_result['write_time']/num_records*1000:.1f}ms/record)"
        )
        print(
            f"  • Read:  {best_result['read_time']:.3f}s ({best_result['read_time']/num_records*1000:.1f}ms/record)"
        )
        print(f"  • Size:  {best_result['compressed_size_mb']:.2f}MB")
        print(
            f"  • Space Saved: {best_result['space_saved_mb']:.2f}MB ({best_result['compression_ratio']*100:.1f}%)"
        )

        print(f"\nBest Compression ({best_space[0].upper()}):")
        best_space_result = best_space[1]
        print(f"  • Compression: {best_space_result['compression_ratio']*100:.1f}%")
        print(f"  • Space Saved: {best_space_result['space_saved_mb']:.2f}MB")

        # Performance comparison
        write_speedup = json_result["write_time"] / best_result["write_time"]
        read_speedup = json_result["read_time"] / best_result["read_time"]

        if write_speedup > 1:
            print(f"  • Write Performance: {write_speedup:.1f}x faster than JSON")
        else:
            print(f"  • Write Performance: {1/write_speedup:.1f}x slower than JSON")

        if read_speedup > 1:
            print(f"  • Read Performance: {read_speedup:.1f}x faster than JSON")
        else:
            print(f"  • Read Performance: {1/read_speedup:.1f}x slower than JSON")

    print("\n🎉 RECOMMENDATIONS")
    print("=" * 60)
    print("Based on the benchmarks:")
    print("• For best compression: Use LZMA with JSON serialization")
    print("• For best performance: Use GZIP with JSON serialization")
    print("• For balanced approach: Use ZLIB with JSON serialization")
    print("• JSON serialization provides good compatibility")
    print("• MessagePack serialization offers better performance (if available)")

    return results


def demonstrate_migration():
    """Demonstrate migration from JSON to compressed storage."""
    print("\n🔄 MIGRATION DEMONSTRATION")
    print("=" * 60)

    # Create some sample JSON metadata
    print("📁 Creating sample JSON metadata...")
    json_storage = MetadataStorage(storage_path="demo_json_metadata", backend="json")

    # Add sample data
    sample_models = ["credit_model", "hiring_model", "medical_model"]
    sample_stages = ["data_ingestion", "training", "validation", "inference"]

    for model in sample_models:
        for stage in sample_stages:
            metadata = generate_sample_metadata(500)
            json_storage.save_metadata(
                model_name=model, stage=stage, event_type="demo", metadata=metadata
            )

    print("✅ Sample JSON metadata created")

    # Create compressed storage and migrate
    print("\n📦 Creating compressed storage...")
    compressed_storage = CompressedMetadataStorage(
        storage_path="demo_compressed_metadata",
        compression="lzma",
        serialization="json",
    )

    print("🔄 Migrating data...")
    migrated_count = compressed_storage.migrate_from_json("demo_json_metadata")

    if migrated_count > 0:
        stats = compressed_storage.get_compression_stats()
        print(f"✅ Successfully migrated {migrated_count} files")
        print(f"📈 Compression ratio: {stats['compression_ratio']*100:.1f}%")
        print(f"💾 Space saved: {stats['space_saved_mb']:.2f} MB")
        print(f"📊 Total files: {stats['total_files']}")
        print(
            f"📏 Original size: {stats['total_uncompressed_size'] / (1024*1024):.2f} MB"
        )
        print(
            f"🗜️ Compressed size: {stats['total_compressed_size'] / (1024*1024):.2f} MB"
        )
    else:
        print("❌ Migration failed")


def main():
    """Main demo function."""
    print("🚀 Starting CIAF Compressed Metadata Storage Demo...")
    print()

    try:
        # Run performance benchmark
        benchmark_storage_performance()

        # Demonstrate migration
        demonstrate_migration()

        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("• Use CompressedMetadataStorage in your projects")
        print("• Migrate existing JSON metadata using tools/metadata/migration.py")
        print("• Choose compression algorithm based on your priorities")

    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
