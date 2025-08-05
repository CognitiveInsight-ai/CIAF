# CIAF: Cognitive Insight AI Framework

[![PyPI - Version](https://img.shields.io/pypi/v/ciaf.svg)](https://pypi.org/project/ciaf/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/CognitiveInsight-ai/CIAF/ci.yml?branch=main)](https://github.com/CognitiveInsight-ai/CIAF/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/ciaf/badge/?version=latest)](https://ciaf.readthedocs.io/en/latest/?badge=latest)

## Overview

The **Cognitive Insight AI Framework (CIAF)** is a production-ready Python framework for **verifiable AI transparency, provenance tracking, and compliance monitoring**. CIAF enables organizations to build trustworthy AI systems with comprehensive audit trails, cryptographic integrity verification, and automated compliance reporting.

## Key Features

### 🔐 Cryptographic Security
- **End-to-end integrity verification** with cryptographic hashing
- **Digital signatures** for tamper-proof audit trails  
- **Key management** for secure operations

### 📊 Data & Model Provenance
- **Dataset anchoring** with cryptographic fingerprinting
- **Complete lineage tracking** from data to decisions
- **Version control** for models and datasets

### 🏛️ Compliance Automation
- **Multi-framework support**: EU AI Act, NIST AI RMF, GDPR, HIPAA, SOX, ISO 27001
- **Automated report generation** with evidence collection
- **Real-time compliance monitoring** and validation

### ⚡ Production Performance
- **Lazy materialization** for 1000x+ performance improvements
- **Compressed metadata storage** with 60-80% space savings
- **Scalable architecture** for enterprise deployments

### 🔧 Easy Integration
- **Framework-agnostic** - works with scikit-learn, TensorFlow, PyTorch, etc.
- **Decorator-based** metadata capture with minimal code changes
- **CLI tools** for setup and compliance reporting

## Installation

### Basic Installation
```bash
pip install ciaf
```

### With Optional Dependencies
```bash
# For compliance reporting
pip install ciaf[compliance]

# For web dashboards  
pip install ciaf[web]

# For advanced visualizations
pip install ciaf[viz]

# Everything
pip install ciaf[dev,compliance,web,viz]
```

### Development Installation
```bash
git clone https://github.com/CognitiveInsight-ai/CIAF.git
cd CIAF
pip install -e .
pip install -r requirements-dev.txt
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

CIAF's true lazy capsule materialization provides exceptional performance:

### Performance Benchmarks (Updated Implementation)
- **True Lazy Approach**: 0.019s for 1,000 items | 0.123s for 10,000 items
- **Eager Approach**: 13.58s for 1,000 items | 137.33s for 10,000 items  
- **Performance Gain**: **703x to 1,113x speedup**

### Real-World Efficiency
- **Memory Usage**: 99% of items remain as lightweight references
- **Selective Auditing**: 95x efficiency for typical audit workflows (processing 1% of dataset)
- **Resource Conservation**: 90% memory reduction for unmaterialized items

### Implementation Modes
```python
# Legacy compatible mode
manager = LazyProvenanceManager(use_true_lazy=False)

# High-performance true lazy mode (recommended)
manager = LazyProvenanceManager(use_true_lazy=True)
```

This makes CIAF suitable for production environments with massive datasets while maintaining full cryptographic integrity.

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