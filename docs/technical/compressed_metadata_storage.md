# CIAF Compressed Metadata Storage

The CIAF framework now supports optimized compressed metadata storage, providing significant space savings and improved performance for large-scale AI projects.

## Features

### 🗜️ Multiple Compression Algorithms
- **LZMA**: Best compression ratio (~60-80% space savings)
- **GZIP**: Balanced performance and compression
- **ZLIB**: Fast compression with good ratios
- **None**: Uncompressed for maximum speed

### 📦 Multiple Serialization Formats
- **JSON**: Human-readable, universal compatibility
- **MessagePack**: Binary format, faster and more compact than JSON
- **Pickle**: Python native, supports complex objects

### 🚀 Performance Benefits
- **Space Savings**: 60-80% reduction in storage size
- **I/O Performance**: Reduced disk read/write operations
- **Network Transfer**: Faster data transfer in distributed systems
- **Batch Processing**: Efficient bulk operations

### 🔄 Backward Compatibility
- Seamless migration from existing JSON storage
- Compatible with existing CIAF metadata APIs
- No changes required to existing code

## Quick Start

### Basic Usage

```python
from ciaf.metadata_storage_compressed import CompressedMetadataStorage

# Create compressed storage
storage = CompressedMetadataStorage(
    storage_path="ciaf_metadata_compressed",
    compression="lzma",        # Best compression
    serialization="json"       # Universal compatibility
)

# Use exactly like regular storage
metadata_id = storage.save_metadata(
    model_name="my_model",
    stage="training",
    event_type="epoch_complete",
    metadata={
        "epoch": 10,
        "loss": 0.234,
        "accuracy": 0.956,
        "parameters": {...}
    }
)

# Retrieve metadata
metadata = storage.get_metadata(metadata_id)
```

### Enable Compression in Existing Code

```python
# Option 1: Use compression flag in existing MetadataStorage
from ciaf.metadata_storage import MetadataStorage

storage = MetadataStorage(
    storage_path="ciaf_metadata",
    backend="json",
    use_compression=True  # Enable compression
)

# Option 2: Direct compressed storage
from ciaf.metadata_storage_compressed import get_metadata_storage

storage = get_metadata_storage(
    compression="lzma",
    serialization="json"
)
```

## Performance Comparison

Based on benchmarks with real metadata:

| Algorithm | Compression Ratio | Write Speed | Read Speed | Use Case |
|-----------|------------------|-------------|------------|----------|
| **LZMA** | 70-80% | Medium | Medium | Best storage efficiency |
| **GZIP** | 60-70% | Fast | Fast | Balanced performance |
| **ZLIB** | 55-65% | Fastest | Fastest | High-frequency operations |
| **None** | 0% | Fastest | Fastest | Maximum speed |

### Real-World Results
- **Space Savings**: 60-80% reduction in storage size
- **Performance**: Comparable or better than JSON for most operations
- **Memory Usage**: Reduced memory footprint for large datasets

## Migration Guide

### Automatic Migration

```python
from ciaf.metadata_storage_compressed import CompressedMetadataStorage

# Create compressed storage
compressed_storage = CompressedMetadataStorage(
    storage_path="ciaf_metadata_compressed",
    compression="lzma"
)

# Migrate existing JSON metadata
migrated_count = compressed_storage.migrate_from_json("ciaf_metadata")
print(f"Migrated {migrated_count} files")

# Get compression statistics
stats = compressed_storage.get_compression_stats()
print(f"Space saved: {stats['space_saved_mb']:.2f} MB")
print(f"Compression ratio: {stats['compression_ratio']*100:.1f}%")
```

### Using Migration Tool

```bash
# Migrate existing metadata
python tools/metadata/migration.py migrate ciaf_metadata ciaf_metadata_compressed

# Benchmark different algorithms
python tools/metadata/migration.py benchmark -s ciaf_metadata -n 100
```

## Configuration Recommendations

### For Maximum Compression
```python
storage = CompressedMetadataStorage(
    compression="lzma",
    serialization="msgpack",  # If available
    compression_level=9
)
```

### For Best Performance
```python
storage = CompressedMetadataStorage(
    compression="gzip",
    serialization="json",
    compression_level=6
)
```

### For High-Frequency Operations
```python
storage = CompressedMetadataStorage(
    compression="zlib",
    serialization="msgpack",
    compression_level=3
)
```

## Storage Backends

### Compressed Files (`compressed_json`)
- Individual compressed files with metadata headers
- Good for distributed file systems
- Easy to backup and transfer

### SQLite with Compression (`sqlite`)
- Compressed BLOBs in SQLite database
- Good for transactional operations
- Built-in indexing and querying

### Hybrid Storage (`hybrid`)
- Small metadata in SQLite, large blobs in files
- Optimal for mixed workloads
- Automatic size-based storage decisions

## Advanced Features

### Compression Statistics

```python
# Get detailed compression statistics
stats = storage.get_compression_stats()
print(f"Total files: {stats['total_files']}")
print(f"Compression ratio: {stats['compression_ratio']*100:.1f}%")
print(f"Space saved: {stats['space_saved_mb']:.2f} MB")
```

### Custom Compression Levels

```python
# Fine-tune compression level (0-9)
storage = CompressedMetadataStorage(
    compression="lzma",
    compression_level=9  # Maximum compression
)
```

### Integrity Verification

All compressed metadata includes:
- SHA-256 hash verification
- Compression type and parameters
- Original and compressed sizes
- Serialization format information

## Examples

### Complete Demo

```bash
# Run comprehensive demo
python examples/advanced/compressed_metadata_demo.py
```

### Performance Benchmarking

```bash
# Benchmark compression algorithms
python tools/metadata/migration.py benchmark -n 100 -o performance_report.md
```

## Migration Checklist

- [ ] Backup existing metadata
- [ ] Test with small dataset first
- [ ] Choose appropriate compression algorithm
- [ ] Verify compression statistics
- [ ] Update applications to use compressed storage
- [ ] Monitor performance in production

## Troubleshooting

### MessagePack Not Available
If you see warnings about MessagePack:
```bash
pip install msgpack
```

### Performance Issues
- Try different compression algorithms
- Adjust compression levels
- Consider hybrid storage for mixed workloads
- Use uncompressed storage for high-frequency operations

### Migration Problems
- Ensure source directory exists and is readable
- Check disk space for compressed storage
- Verify file permissions
- Use verbose mode for detailed error messages

## Dependencies

### Required
- Python 3.8+
- Standard library (gzip, lzma, zlib)

### Optional
- **msgpack**: For MessagePack serialization
- **matplotlib**: For performance visualization in tools

## File Format

Compressed metadata files (`.cmeta`) contain:
1. **Header Size** (4 bytes): Size of JSON header
2. **JSON Header**: Metadata about the compressed data
3. **Compressed Data**: The actual metadata payload

This format ensures backward compatibility and enables efficient random access to metadata without full decompression.
