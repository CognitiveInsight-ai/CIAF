# CIAF Examples

This directory contains examples and demonstrations of CIAF capabilities organized by complexity and use case.

## Directory Structure

### [`basic/`](basic/)
Getting started examples for new users:
- [`quick_start.py`](basic/quick_start.py) - Minimal CIAF setup with scikit-learn
- [`sklearn_example.py`](basic/sklearn_example.py) - Complete sklearn integration example
- [`practical_example.py`](basic/practical_example.py) - Real-world usage patterns
- [`pre_ingestion_demo.py`](basic/pre_ingestion_demo.py) - Data validation and preprocessing

### [`compliance/`](compliance/)
Regulatory compliance demonstrations:
- [`eu_ai_act_demo.py`](compliance/eu_ai_act_demo.py) - EU AI Act compliance for high-risk systems
- [`compliance_demo_simple.py`](compliance/compliance_demo_simple.py) - Basic compliance setup
- [`compliance_demo_comprehensive.py`](compliance/compliance_demo_comprehensive.py) - Multi-framework compliance
- [`ciaf_360_compliance_demo.py`](compliance/ciaf_360_compliance_demo.py) - Complete governance demo

### [`industry/`](industry/)
Industry-specific use cases:

#### [`healthcare/`](industry/healthcare/)
- Medical diagnosis systems
- CT scan classification
- HIPAA compliance examples

#### [`finance/`](industry/finance/)
- Credit scoring models
- Loan approval systems
- Financial services compliance

#### [`hiring/`](industry/hiring/)
- Job classification systems
- Bias detection and mitigation
- Fair hiring practices

### [`advanced/`](advanced/)
Advanced features and customization:
- [`ciaf_comprehensive_demo.py`](advanced/ciaf_comprehensive_demo.py) - Full framework capabilities
- [`best_practices_example.py`](advanced/best_practices_example.py) - Production deployment patterns
- Custom compliance frameworks
- Performance optimization techniques

## Quick Start

1. **Basic Usage**: Start with [`basic/quick_start.py`](basic/quick_start.py)
2. **Compliance**: Try [`compliance/eu_ai_act_demo.py`](compliance/eu_ai_act_demo.py)
3. **Industry Examples**: Explore [`industry/`](industry/) for domain-specific examples

## Running Examples

```bash
# From the CIAF root directory
cd examples

# Run basic example
python basic/quick_start.py

# Run compliance demo
python compliance/eu_ai_act_demo.py

# Run industry-specific example
python industry/finance/credit_scoring.py
```

## Requirements

Most examples require the basic CIAF installation:

```bash
pip install ciaf
```

Some advanced examples may require additional dependencies:

```bash
pip install -r ../requirements-dev.txt
```

## Contributing Examples

When adding new examples:

1. Place in the appropriate subdirectory based on complexity/topic
2. Include comprehensive docstrings and comments
3. Follow the established naming convention
4. Add entry to this README
5. Ensure examples are self-contained and runnable

## Support

For questions about examples:
- Check the [documentation](../docs/)
- Open an issue on GitHub
- See the main [README](../README.md) for contact information
