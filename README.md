# CIAF: Cognitive Insight AI Framework

[![PyPI - Version](https://img.shields.io/pypi/v/ciaf.svg)](https://pypi.org/project/ciaf/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/your-github-username/ciaf/main.yml?branch=main)](https://github.com/your-github-username/ciaf/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/ciaf/badge/?version=latest)](https://ciaf.readthedocs.io/en/latest/?badge=latest)

## Overview

The **Cognitive Insight AI Framework (CIAF)** is a modular Python framework designed to provide **verifiable transparency, provenance tracking, and cryptographic integrity for Artificial Intelligence (AI) systems** throughout their development and deployment lifecycle. Built with a focus on modularity and performance, CIAF enables developers to add comprehensive audit trails and verification capabilities to their AI workflows.

## Key Features

CIAF provides a comprehensive chain of verifiable trust through its modular architecture:

* **🔐 Cryptographic Core:** Robust encryption, key derivation, and hash-based integrity verification
* **📊 Dataset Anchoring:** Secure dataset fingerprinting with lazy capsule materialization for optimal performance
* **🔗 Provenance Tracking:** Complete lineage tracking from data ingestion through model training to inference
* **🧠 ML Integration:** Seamless integration with machine learning frameworks and model wrappers
* **⚡ Performance Optimized:** Lazy materialization providing 29,000x+ performance improvements over eager approaches
* **🗜️ Compressed Storage:** Optimized metadata storage with 60-80% space savings and improved I/O performance
* **🏗️ Modular Design:** Clean separation of concerns with independent, composable modules

## Installation

You can install `ciaf` using pip:

```bash
pip install ciaf
```

Or for development:

```bash
git clone https://github.com/CognitiveInsight-ai/CIAF.git
cd CIAF
make install-dev
```

Alternatively, you can use the setup script:

```bash
python scripts/setup_dev_env.py
```

## Quick Examples

### Basic Model Wrapper
```python
from ciaf.wrappers import CIAFModelWrapper
from sklearn.linear_model import LogisticRegression

# Your existing model
model = LogisticRegression()
wrapped_model = CIAFModelWrapper(model, model_id="my_model")

# Use normally - now with full CIAF tracking
wrapped_model.fit(X_train, y_train)
predictions = wrapped_model.predict(X_test)
```

### Compliance Monitoring
```python
from ciaf.compliance import ComplianceTracker, ComplianceFramework

tracker = ComplianceTracker()
tracker.track_compliance(
    model_id="my_model",
    frameworks=[ComplianceFramework.EU_AI_ACT, ComplianceFramework.GDPR]
)
```

More examples available in the [`examples/`](examples/) directory:
- [`examples/basic/`](examples/basic/) - Getting started examples
- [`examples/compliance/`](examples/compliance/) - Regulatory compliance demos  
- [`examples/industry/`](examples/industry/) - Industry-specific use cases
- [`examples/advanced/`](examples/advanced/) - Advanced features and compressed metadata storage

## Development

### Metadata Migration
To upgrade existing JSON metadata to compressed format for better performance:
```bash
python tools/metadata/migration.py --source-dir ciaf_metadata --target-dir ciaf_metadata_compressed
```

### Running Tests
```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests only  
make test-integration
```

### Code Quality
```bash
# Format code
make format

# Run linting
make lint
```

### Documentation
```bash
# Build documentation
make docs
```
model = LogisticRegression()

# Wrap with CIAF
wrapped_model = CIAFModelWrapper(
    model=model,
    model_name="MyLogisticRegression"
)

# Your training data
training_data = [
    {"content": "example 1", "metadata": {"target": 0}},
    {"content": "example 2", "metadata": {"target": 1}},
    # ... more data
]

# Train with automatic CIAF tracking
snapshot = wrapped_model.train(
    dataset_id="my_dataset",
    training_data=training_data,
    master_password="secure_password",
    model_version="1.0.0"
)

# Make predictions with verification receipts
prediction, receipt = wrapped_model.predict("test input")
print(f"Prediction: {prediction}")
print(f"Receipt ID: {receipt.receipt_id}")
```

### Full Framework Usage

Here's a more detailed example using the complete CIAF API:

```python
from ciaf.api import CIAFFramework
from ciaf.provenance import ProvenanceCapsule
from ciaf.anchoring import DatasetAnchor
from ciaf.inference import InferenceReceipt

# Initialize framework
framework = CIAFFramework("MyAIProject")

# Create dataset anchor
dataset_metadata = {
    "name": "Customer Reviews Dataset",
    "version": "1.0",
    "description": "Product reviews for sentiment analysis"
}

anchor = framework.create_dataset_anchor(
    dataset_id="reviews_v1",
    dataset_metadata=dataset_metadata,
    master_password="secure_password"
)

# Add training data with provenance
training_data = [
    {"content": "Great product!", "metadata": {"sentiment": "positive", "id": "rev_001"}},
    {"content": "Poor quality", "metadata": {"sentiment": "negative", "id": "rev_002"}},
    # ... more data
]

# Create provenance capsules (lazy materialization)
capsules = framework.create_provenance_capsules("reviews_v1", training_data)

print(f"Created {len(capsules)} provenance capsules")
print(f"Dataset fingerprint: {anchor.dataset_fingerprint}")
```

## Architecture

CIAF is organized into six core modules:

### Core (`ciaf.core`)
- **CryptoUtils**: Encryption, hashing, and cryptographic primitives
- **KeyManager**: Key derivation and management
- **MerkleTree**: Merkle tree implementation for integrity verification

### Anchoring (`ciaf.anchoring`)
- **DatasetAnchor**: Dataset fingerprinting and key derivation
- **LazyManager**: Lazy capsule materialization for performance optimization

### Provenance (`ciaf.provenance`)
- **ProvenanceCapsule**: Individual data item provenance tracking
- **ModelAggregationKey**: Model-level cryptographic keys
- **TrainingSnapshot**: Complete training session snapshots

### Inference (`ciaf.inference`)
- **InferenceReceipt**: Individual prediction receipts
- **ZKEChain**: Chained inference receipt management

### Simulation (`ciaf.simulation`)
- **MLFrameworkSimulator**: ML framework simulation and testing
- **MockLLM**: Large Language Model simulation

### Wrappers (`ciaf.wrappers`)
- **CIAFModelWrapper**: Drop-in wrapper for existing ML models

### API (`ciaf.api`)
- **CIAFFramework**: Main framework orchestration class

## Performance

CIAF's lazy capsule materialization provides exceptional performance:

- **Lazy Approach**: ~0.006 seconds for 1000 items
- **Eager Approach**: ~179 seconds for 1000 items
- **Performance Gain**: 29,361x speedup

This makes CIAF suitable for production environments with large datasets.

## Documentation

- **Architecture Guide**: [buildDocs/lazy_capsule_materialization.md](buildDocs/lazy_capsule_materialization.md)
- **API Reference**: Coming soon
- **Examples**: See `ciaf_comprehensive_demo.py` for complete usage examples

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

Or run the comprehensive demo:

```bash
python ciaf_comprehensive_demo.py
```

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or feature requests, please visit our [GitHub Issues](https://github.com/your-github-username/ciaf/issues) page.

---

**CIAF: Bringing verifiable transparency to AI systems, one module at a time.**