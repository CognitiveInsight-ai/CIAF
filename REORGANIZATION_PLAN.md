# CIAF Codebase Reorganization Plan

## Current Issues Identified

### 1. Root Directory Clutter
- Too many demo/test files in root directory
- Mixed documentation locations
- Inconsistent naming conventions

### 2. Directory Structure Problems
- Multiple metadata directories with unclear purposes
- Test files scattered across multiple locations
- Examples and demos not properly organized

### 3. Build Management
- Build artifacts in version control
- Missing proper .gitignore entries

## Proposed New Structure

```
CIAF/
├── .github/                          # GitHub workflows and templates
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── release.yml
│   │   └── docs.yml
│   └── ISSUE_TEMPLATE/
├── .gitignore                        # Comprehensive gitignore
├── README.md                         # Main project README
├── CHANGELOG.md                      # Keep at root
├── LICENSE                           # Keep at root
├── pyproject.toml                    # Modern Python packaging
├── requirements.txt                  # Basic requirements
├── requirements-dev.txt              # Development requirements
├── Makefile                          # Build automation
├── ciaf/                             # Main package (keep current structure)
│   ├── __init__.py
│   ├── core/
│   ├── anchoring/
│   ├── provenance/
│   ├── inference/
│   ├── simulation/
│   ├── wrappers/
│   ├── api/
│   ├── compliance/
│   ├── explainability/
│   ├── uncertainty/
│   ├── preprocessing/
│   └── metadata_tags/
├── tests/                            # Consolidated test directory
│   ├── __init__.py
│   ├── conftest.py                   # pytest configuration
│   ├── unit/                         # Unit tests
│   │   ├── test_core.py
│   │   ├── test_anchoring.py
│   │   ├── test_provenance.py
│   │   └── ...
│   ├── integration/                  # Integration tests
│   │   ├── test_full_pipeline.py
│   │   ├── test_compliance_workflow.py
│   │   └── ...
│   ├── performance/                  # Performance tests
│   │   ├── test_lazy_loading.py
│   │   └── benchmark_suite.py
│   └── fixtures/                     # Test data and fixtures
├── examples/                         # All examples and demos
│   ├── __init__.py
│   ├── basic/                        # Basic usage examples
│   │   ├── quick_start.py
│   │   ├── model_wrapper_demo.py
│   │   └── simple_pipeline.py
│   ├── compliance/                   # Compliance-focused examples
│   │   ├── eu_ai_act_demo.py
│   │   ├── nist_ai_rmf_demo.py
│   │   ├── gdpr_compliance.py
│   │   └── comprehensive_compliance.py
│   ├── industry/                     # Industry-specific examples
│   │   ├── healthcare/
│   │   │   ├── ct_scan_classifier.py
│   │   │   └── medical_diagnosis.py
│   │   ├── finance/
│   │   │   ├── credit_scoring.py
│   │   │   └── loan_approval.py
│   │   └── hiring/
│   │       ├── job_classifier.py
│   │       └── bias_detection.py
│   ├── advanced/                     # Advanced usage examples
│   │   ├── custom_compliance.py
│   │   ├── visualization_demo.py
│   │   └── performance_optimization.py
│   └── notebooks/                    # Jupyter notebooks
│       ├── Getting_Started.ipynb
│       ├── Compliance_Tutorial.ipynb
│       └── Performance_Analysis.ipynb
├── docs/                             # All documentation
│   ├── source/                       # Sphinx source
│   │   ├── conf.py
│   │   ├── index.rst
│   │   ├── api/
│   │   ├── tutorials/
│   │   └── compliance/
│   ├── build/                        # Built documentation (gitignored)
│   ├── technical/                    # Technical documentation
│   │   ├── architecture.md
│   │   ├── performance.md
│   │   └── security.md
│   ├── compliance/                   # Compliance documentation
│   │   ├── frameworks/
│   │   ├── implementation_guide.md
│   │   └── audit_preparation.md
│   ├── legal/                        # Legal and patent docs
│   │   ├── patents/
│   │   ├── licensing.md
│   │   └── compliance_statements.md
│   └── media/                        # Images, diagrams, etc.
├── tools/                            # Development and utility tools
│   ├── __init__.py
│   ├── metadata/                     # Metadata management tools
│   │   ├── viewer.py
│   │   ├── export.py
│   │   └── validation.py
│   ├── compliance/                   # Compliance tools
│   │   ├── report_generator.py
│   │   ├── framework_validator.py
│   │   └── audit_helper.py
│   ├── performance/                  # Performance tools
│   │   ├── benchmarks.py
│   │   └── profiler.py
│   └── deployment/                   # Deployment tools
│       ├── docker/
│       ├── kubernetes/
│       └── cloud/
├── scripts/                          # Build and automation scripts
│   ├── setup_dev_env.py
│   ├── run_tests.py
│   ├── build_docs.py
│   ├── release.py
│   └── clean.py
├── data/                             # Sample and test data
│   ├── __init__.py
│   ├── samples/                      # Sample datasets
│   ├── templates/                    # Configuration templates
│   └── schemas/                      # Data schemas
└── deployments/                      # Deployment configurations
    ├── docker/
    │   ├── Dockerfile
    │   ├── docker-compose.yml
    │   └── README.md
    ├── kubernetes/
    │   ├── manifests/
    │   └── helm/
    └── cloud/
        ├── aws/
        ├── azure/
        └── gcp/
```

## Migration Steps

### Phase 1: Core Structure Setup
1. Create new directory structure
2. Update .gitignore to exclude build artifacts
3. Create development requirements file

### Phase 2: Code Organization
1. Move all demo files to appropriate example directories
2. Consolidate test files into unified test structure
3. Organize documentation into docs/ hierarchy

### Phase 3: Build System Updates
1. Update pyproject.toml with proper build configuration
2. Create Makefile for common development tasks
3. Set up GitHub Actions for CI/CD

### Phase 4: Documentation Reorganization
1. Move all documentation to docs/ structure
2. Set up Sphinx for documentation generation
3. Organize compliance documentation

### Phase 5: Tooling and Utilities
1. Create utility scripts for common tasks
2. Set up development environment scripts
3. Create deployment configurations

## Benefits of This Reorganization

### 1. Improved Developer Experience
- Clear separation of concerns
- Easy to find examples and documentation
- Standardized development workflow

### 2. Better Maintenance
- Centralized testing structure
- Organized documentation
- Clear build process

### 3. Enhanced Compliance
- Dedicated compliance documentation
- Structured audit trail
- Clear example implementations

### 4. Professional Structure
- Industry-standard organization
- Easy onboarding for new developers
- Clear contribution guidelines

## Implementation Priority

1. **High Priority**: Core structure, build system, test organization
2. **Medium Priority**: Documentation reorganization, example organization
3. **Low Priority**: Advanced tooling, deployment configurations

## Backward Compatibility

- All existing imports will continue to work
- No breaking changes to the main ciaf package
- Migration can be done incrementally
