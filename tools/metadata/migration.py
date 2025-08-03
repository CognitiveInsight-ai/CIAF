#!/usr/bin/env python3
"""
CIAF Metadata Storage Migration and Performance Tool

Tool to migrate from JSON to compressed storage and compare performance.
"""

import time
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from datetime import datetime

# Add CIAF to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from ciaf.metadata_storage import MetadataStorage
    from ciaf.metadata_storage_compressed import CompressedMetadataStorage
except ImportError as e:
    print(f"❌ Could not import CIAF modules: {e}")
    sys.exit(1)


class MetadataStorageAnalyzer:
    """Analyze and compare metadata storage performance."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_compression_algorithms(
        self, 
        source_path: str,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark different compression algorithms.
        
        Args:
            source_path: Path to existing JSON metadata
            sample_size: Number of files to test
            
        Returns:
            Performance comparison results
        """
        print("🔬 Benchmarking Compression Algorithms...")
        
        # Load sample data
        sample_data = self._load_sample_data(source_path, sample_size)
        if not sample_data:
            print("❌ No sample data found")
            return {}
        
        print(f"📊 Testing with {len(sample_data)} metadata records")
        
        compression_algorithms = ["none", "gzip", "lzma", "zlib"]
        serialization_formats = ["json", "pickle"]
        
        # Add msgpack if available
        try:
            import msgpack
            serialization_formats.append("msgpack")
        except ImportError:
            pass
        
        results = {}
        
        for compression in compression_algorithms:
            for serialization in serialization_formats:
                print(f"🧪 Testing {compression}/{serialization}...")
                
                try:
                    storage = CompressedMetadataStorage(
                        storage_path=f"benchmark_{compression}_{serialization}",
                        compression=compression,
                        serialization=serialization
                    )
                    
                    # Benchmark write performance
                    start_time = time.time()
                    metadata_ids = []
                    
                    for i, data in enumerate(sample_data):
                        metadata_id = storage.save_metadata(
                            model_name=f"benchmark_model_{i}",
                            stage="benchmark",
                            event_type="test",
                            metadata=data
                        )
                        metadata_ids.append(metadata_id)
                    
                    write_time = time.time() - start_time
                    
                    # Benchmark read performance
                    start_time = time.time()
                    
                    for metadata_id in metadata_ids:
                        storage.get_metadata(metadata_id)
                    
                    read_time = time.time() - start_time
                    
                    # Get compression stats
                    stats = storage.get_compression_stats()
                    
                    key = f"{compression}/{serialization}"
                    results[key] = {
                        "write_time": write_time,
                        "read_time": read_time,
                        "total_time": write_time + read_time,
                        "compression_ratio": stats["compression_ratio"],
                        "space_saved_mb": stats["space_saved_mb"],
                        "files_processed": stats["total_files"],
                        "avg_write_time": write_time / len(sample_data),
                        "avg_read_time": read_time / len(sample_data)
                    }
                    
                    print(f"   ✅ Write: {write_time:.3f}s, Read: {read_time:.3f}s, "
                          f"Compression: {stats['compression_ratio']*100:.1f}%")
                    
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    continue
        
        self.results = results
        return results
    
    def _load_sample_data(self, source_path: str, sample_size: int) -> List[Dict[str, Any]]:
        """Load sample metadata for testing."""
        source_dir = Path(source_path)
        sample_data = []
        
        if not source_dir.exists():
            # Generate synthetic data if no source available
            return self._generate_synthetic_data(sample_size)
        
        for model_dir in source_dir.iterdir():
            if model_dir.is_dir():
                for json_file in model_dir.glob("*.json"):
                    if len(sample_data) >= sample_size:
                        break
                    
                    try:
                        with open(json_file, 'r') as f:
                            record = json.load(f)
                        sample_data.append(record.get("metadata", {}))
                    except Exception:
                        continue
                
                if len(sample_data) >= sample_size:
                    break
        
        return sample_data
    
    def _generate_synthetic_data(self, size: int) -> List[Dict[str, Any]]:
        """Generate synthetic metadata for testing."""
        import random
        import string
        
        synthetic_data = []
        
        for i in range(size):
            data = {
                "id": f"synthetic_{i}",
                "timestamp": datetime.now().isoformat(),
                "features": [f"feature_{j}" for j in range(random.randint(10, 50))],
                "parameters": {
                    f"param_{j}": random.random() for j in range(random.randint(5, 20))
                },
                "metrics": {
                    "accuracy": random.random(),
                    "precision": random.random(),
                    "recall": random.random(),
                    "f1_score": random.random()
                },
                "training_data": {
                    "samples": random.randint(1000, 10000),
                    "features": random.randint(10, 100),
                    "classes": random.randint(2, 10)
                },
                "description": ''.join(random.choices(string.ascii_letters + ' ', k=random.randint(100, 500))),
                "tags": [f"tag_{j}" for j in range(random.randint(3, 10))],
                "nested_data": {
                    "level1": {
                        "level2": {
                            "data": [random.random() for _ in range(random.randint(10, 100))]
                        }
                    }
                }
            }
            synthetic_data.append(data)
        
        return synthetic_data
    
    def create_performance_report(self, output_path: str = None) -> str:
        """Create a detailed performance report."""
        if not self.results:
            return "No benchmark results available"
        
        # Sort results by total time
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]["total_time"]
        )
        
        report = []
        report.append("# CIAF Metadata Storage Performance Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Summary table
        report.append("## Performance Summary")
        report.append("")
        report.append("| Algorithm | Write Time (s) | Read Time (s) | Total Time (s) | Compression % | Space Saved (MB) |")
        report.append("|-----------|----------------|---------------|----------------|---------------|------------------|")
        
        for key, result in sorted_results:
            report.append(
                f"| {key} | {result['write_time']:.3f} | {result['read_time']:.3f} | "
                f"{result['total_time']:.3f} | {result['compression_ratio']*100:.1f}% | "
                f"{result['space_saved_mb']:.2f} |"
            )
        
        report.append("")
        
        # Best performing configurations
        best_write = min(sorted_results, key=lambda x: x[1]["write_time"])
        best_read = min(sorted_results, key=lambda x: x[1]["read_time"])
        best_compression = max(sorted_results, key=lambda x: x[1]["compression_ratio"])
        
        report.append("## Recommendations")
        report.append("")
        report.append(f"**Fastest Write**: {best_write[0]} ({best_write[1]['write_time']:.3f}s)")
        report.append(f"**Fastest Read**: {best_read[0]} ({best_read[1]['read_time']:.3f}s)")
        report.append(f"**Best Compression**: {best_compression[0]} ({best_compression[1]['compression_ratio']*100:.1f}%)")
        report.append("")
        
        # Detailed analysis
        report.append("## Detailed Analysis")
        report.append("")
        
        for key, result in sorted_results:
            report.append(f"### {key}")
            report.append(f"- **Write Performance**: {result['avg_write_time']*1000:.3f}ms per record")
            report.append(f"- **Read Performance**: {result['avg_read_time']*1000:.3f}ms per record")
            report.append(f"- **Compression Ratio**: {result['compression_ratio']*100:.1f}%")
            report.append(f"- **Space Savings**: {result['space_saved_mb']:.2f} MB")
            report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"📊 Performance report saved to: {output_path}")
        
        return report_text
    
    def migrate_metadata(
        self, 
        source_path: str,
        target_path: str,
        compression: str = "lzma",
        serialization: str = "json"
    ) -> int:
        """
        Migrate existing JSON metadata to compressed format.
        
        Args:
            source_path: Path to existing JSON metadata
            target_path: Path for compressed metadata
            compression: Compression algorithm to use
            serialization: Serialization format to use
            
        Returns:
            Number of files migrated
        """
        print(f"🔄 Migrating metadata from {source_path} to {target_path}")
        print(f"📦 Using {compression} compression with {serialization} serialization")
        
        # Create compressed storage
        compressed_storage = CompressedMetadataStorage(
            storage_path=target_path,
            compression=compression,
            serialization=serialization
        )
        
        # Migrate data
        migrated_count = compressed_storage.migrate_from_json(source_path)
        
        if migrated_count > 0:
            stats = compressed_storage.get_compression_stats()
            print(f"✅ Successfully migrated {migrated_count} files")
            print(f"📈 Compression ratio: {stats['compression_ratio']*100:.1f}%")
            print(f"💾 Space saved: {stats['space_saved_mb']:.2f} MB")
        else:
            print("⚠️ No files were migrated")
        
        return migrated_count


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="CIAF Metadata Storage Migration and Performance Tool")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark compression algorithms')
    benchmark_parser.add_argument('-s', '--source', default='ciaf_metadata', help='Source metadata path')
    benchmark_parser.add_argument('-n', '--samples', type=int, default=100, help='Number of samples to test')
    benchmark_parser.add_argument('-o', '--output', help='Output report path')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate JSON metadata to compressed format')
    migrate_parser.add_argument('source', help='Source JSON metadata path')
    migrate_parser.add_argument('target', help='Target compressed metadata path')
    migrate_parser.add_argument('-c', '--compression', default='lzma', 
                               choices=['none', 'gzip', 'lzma', 'zlib'], help='Compression algorithm')
    migrate_parser.add_argument('-f', '--format', default='json',
                               choices=['json', 'msgpack', 'pickle'], help='Serialization format')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    analyzer = MetadataStorageAnalyzer()
    
    if args.command == 'benchmark':
        print("🚀 Starting compression benchmark...")
        results = analyzer.benchmark_compression_algorithms(args.source, args.samples)
        
        if results:
            report = analyzer.create_performance_report(args.output)
            if not args.output:
                print("\n" + report)
        else:
            print("❌ Benchmark failed")
    
    elif args.command == 'migrate':
        migrated = analyzer.migrate_metadata(
            args.source, args.target, args.compression, args.format
        )
        if migrated == 0:
            print("❌ Migration failed or no files found")


if __name__ == "__main__":
    main()
