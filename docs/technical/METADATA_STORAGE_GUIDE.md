# CIAF Metadata Storage System

## Overview

The CIAF (Cognitive Insight AI Framework) Metadata Storage System provides comprehensive metadata capture, storage, and retrieval capabilities for AI/ML pipelines. It enables full traceability of data processing, model training, inference, and compliance monitoring across the entire AI lifecycle.

## Key Features

- **Multiple Storage Backends**: JSON, SQLite, and Pickle formats
- **Automatic Metadata Capture**: Decorators and context managers for seamless integration
- **Compliance Tracking**: Built-in support for GDPR, FDA, EEOC, FCRA, HIPAA, and ISO 13485
- **Pipeline Tracing**: Complete audit trails from data ingestion to inference
- **Export Capabilities**: Multiple export formats for reporting and analysis
- **Configurable Retention**: Flexible data retention policies
- **Performance Optimized**: Designed for high-throughput AI systems

## Architecture

### Core Components

1. **MetadataStorage**: Core storage engine with pluggable backends
2. **MetadataConfig**: Configuration management system
3. **ModelMetadataManager**: High-level API for model lifecycle tracking
4. **ComplianceTracker**: Specialized compliance monitoring
5. **MetadataCapture**: Decorators and context managers for automatic capture

### Storage Backends

#### JSON Backend
- **Best for**: Small to medium projects, development, version control
- **Advantages**: Human-readable, easy to inspect, git-friendly
- **File organization**: Organized by model name and date
- **Use case**: Prototyping, debugging, regulatory documentation

#### SQLite Backend
- **Best for**: Large projects, production systems, complex queries
- **Advantages**: ACID compliance, SQL queries, better performance
- **Features**: Structured queries, joins, indexing
- **Use case**: Production systems, analytics, reporting

#### Pickle Backend
- **Best for**: Complex Python objects, rapid prototyping
- **Advantages**: Native Python serialization, handles any object
- **Limitations**: Python-specific, not human-readable
- **Use case**: Research, complex data structures, Python-only environments

## Getting Started

### 1. Basic Setup

```python
from ciaf.metadata_storage import MetadataStorage
from ciaf.metadata_config import MetadataConfig

# Initialize storage
storage = MetadataStorage("my_project_metadata", backend="json")

# Save metadata
metadata_id = storage.save_metadata(
    model_name="my_classifier",
    stage="training",
    event_type="model_trained",
    metadata={
        "accuracy": 0.95,
        "training_time": 120,
        "algorithm": "RandomForest"
    },
    details="Model training completed successfully"
)
```

### 2. Using ModelMetadataManager

```python
from ciaf.metadata_integration import ModelMetadataManager

# Create manager for your model
manager = ModelMetadataManager("customer_classifier", "2.1.0")

# Log data ingestion
manager.log_data_ingestion({
    "rows": 10000,
    "columns": 25,
    "quality_score": 0.92,
    "source": "customer_database"
})

# Log training
manager.log_training_start({
    "algorithm": "XGBoost",
    "hyperparameters": {"n_estimators": 100, "max_depth": 6}
})

manager.log_training_complete({
    "accuracy": 0.94,
    "precision": 0.92,
    "recall": 0.96,
    "training_time_minutes": 15
})

# Get complete pipeline trace
trace = manager.get_pipeline_trace()
```

### 3. Automatic Metadata Capture

```python
from ciaf.metadata_integration import capture_metadata

@capture_metadata("my_model", "preprocessing", "data_cleaning")
def clean_data(data_path: str, remove_outliers: bool = True):
    """Automatically captures function parameters and performance."""
    # Your data cleaning code here
    return cleaned_data

# Or use context manager
from ciaf.metadata_integration import MetadataCapture

with MetadataCapture("my_model", "inference", "batch_prediction") as capture:
    capture.add_metadata("batch_size", 1000)
    # Your prediction code here
    capture.add_metadata("predictions_made", 1000)
```

### 4. Compliance Tracking

```python
from ciaf.metadata_integration import ComplianceTracker

tracker = ComplianceTracker(manager)

# Track GDPR compliance
tracker.track_gdpr_compliance(
    data_protection_measures={
        "encryption": True,
        "anonymization": True,
        "access_controls": True
    },
    consent_management={
        "explicit_consent": True,
        "withdrawal_mechanism": True
    },
    right_to_explanation=True
)

# Track FDA compliance (medical AI)
tracker.track_fda_compliance(
    clinical_validation={
        "clinical_studies": True,
        "performance_validation": True
    },
    safety_measures={
        "risk_assessment": True,
        "monitoring_plan": True
    },
    quality_management={
        "iso_13485": True,
        "documentation": True
    }
)
```

## Configuration

### Configuration File Example

```json
{
  "storage_backend": "sqlite",
  "storage_path": "/var/lib/myapp/metadata",
  "metadata_retention_days": 365,
  "compliance_retention_days": 2555,
  "auto_cleanup_enabled": true,
  "compliance_frameworks": ["GDPR", "FDA", "EEOC"],
  "enable_metrics": true,
  "audit_all_access": true
}
```

### Environment Variables

```bash
export CIAF_STORAGE_BACKEND=sqlite
export CIAF_STORAGE_PATH=/data/metadata
export CIAF_RETENTION_DAYS=365
export CIAF_AUTO_CLEANUP=true
export CIAF_ENCRYPT_DATA=true
```

### Configuration Templates

```python
from ciaf.metadata_config import create_config_template

# Create production configuration
create_config_template("production", "prod_config.json")

# Create development configuration
create_config_template("development", "dev_config.json")
```

## Pipeline Stages

The metadata system organizes information by pipeline stages:

### 1. Data Ingestion
- Data source information
- Data quality metrics
- Schema validation
- Volume and completeness

### 2. Data Preprocessing
- Transformation steps
- Feature engineering
- Data cleaning operations
- Quality improvements

### 3. Training
- Algorithm selection
- Hyperparameters
- Training performance
- Model artifacts

### 4. Validation
- Test set performance
- Cross-validation results
- Bias assessment
- Fairness metrics

### 5. Deployment
- Deployment configuration
- Infrastructure details
- Performance requirements
- Monitoring setup

### 6. Inference
- Prediction requests
- Response times
- Model versions
- Input/output logging

### 7. Monitoring
- Model drift detection
- Performance degradation
- Alert conditions
- Maintenance events

### 8. Compliance
- Regulatory validations
- Audit events
- Corrective actions
- Compliance scores

## Metadata Schema

### Standard Metadata Fields

```json
{
  "id": "unique-metadata-id",
  "model_name": "customer_classifier",
  "model_version": "2.1.0",
  "stage": "training",
  "event_type": "model_trained",
  "timestamp": "2024-08-02T10:30:00Z",
  "metadata_hash": "sha256-hash",
  "details": "Human-readable description",
  "metadata": {
    // Stage-specific metadata
  }
}
```

### Training Metadata Example

```json
{
  "metadata": {
    "algorithm": "XGBoostClassifier",
    "hyperparameters": {
      "n_estimators": 100,
      "max_depth": 6,
      "learning_rate": 0.1
    },
    "training_samples": 8000,
    "validation_samples": 2000,
    "performance": {
      "accuracy": 0.94,
      "precision": 0.92,
      "recall": 0.96,
      "f1_score": 0.94,
      "auc_roc": 0.97
    },
    "training_time_seconds": 450,
    "feature_importance": [
      {"feature": "age", "importance": 0.23},
      {"feature": "income", "importance": 0.19}
    ]
  }
}
```

### Inference Metadata Example

```json
{
  "metadata": {
    "request_id": "req_123456",
    "input_features": 15,
    "prediction": "approved",
    "confidence": 0.87,
    "processing_time_ms": 12.5,
    "model_version": "2.1.0",
    "feature_values_hash": "sha256-hash"
  }
}
```

## Querying and Retrieval

### Get Specific Metadata

```python
# By ID
metadata = storage.get_metadata("metadata-id-123")

# By model and stage
training_events = storage.get_model_metadata(
    "customer_classifier", 
    stage="training", 
    limit=10
)

# Complete pipeline trace
trace = storage.get_pipeline_trace("customer_classifier")
```

### SQL Queries (SQLite backend)

```python
# Custom queries for advanced analytics
conn = sqlite3.connect(storage.db_path)
cursor = conn.cursor()

# Find models with low accuracy
cursor.execute("""
    SELECT model_name, metadata_json 
    FROM metadata 
    WHERE stage = 'training' 
    AND json_extract(metadata_json, '$.accuracy') < 0.9
""")

# Compliance events in last 30 days
cursor.execute("""
    SELECT framework, compliance_score, timestamp
    FROM compliance_events
    WHERE timestamp > datetime('now', '-30 days')
    ORDER BY timestamp DESC
""")
```

## Export and Reporting

### Export Options

```python
# Export specific model
json_path = storage.export_metadata("my_model", "json")
csv_path = storage.export_metadata("my_model", "csv")
xml_path = storage.export_metadata("my_model", "xml")

# Export all metadata
all_data = storage.export_metadata(format="json")
```

### Compliance Reports

```python
# Generate compliance report
def generate_compliance_report(model_name: str):
    manager = ModelMetadataManager(model_name)
    trace = manager.get_pipeline_trace()
    
    report = {
        "model": model_name,
        "generated": datetime.now().isoformat(),
        "stages": len(trace["stages"]),
        "compliance_events": []
    }
    
    # Add compliance events
    compliance_metadata = manager.get_stage_metadata("compliance")
    for event in compliance_metadata:
        report["compliance_events"].append({
            "framework": event["event_type"],
            "score": event["metadata"].get("compliance_score"),
            "timestamp": event["timestamp"]
        })
    
    return report
```

## Integration with Existing Models

### Minimal Integration

```python
class ExistingModel:
    def __init__(self):
        # Add metadata manager
        self.metadata_manager = ModelMetadataManager("existing_model", "1.0.0")
    
    def train(self, X, y):
        # Log training start
        self.metadata_manager.log_training_start({
            "samples": len(X),
            "features": X.shape[1]
        })
        
        # Your existing training code
        model = self._train_model(X, y)
        
        # Log training completion
        accuracy = self._evaluate_model(model)
        self.metadata_manager.log_training_complete({
            "accuracy": accuracy
        })
        
        return model
```

### Comprehensive Integration

```python
from ciaf.metadata_integration import ModelMetadataManager, ComplianceTracker, capture_metadata

class FullyIntegratedModel:
    def __init__(self, model_name: str):
        self.metadata_manager = ModelMetadataManager(model_name, "2.0.0")
        self.compliance_tracker = ComplianceTracker(self.metadata_manager)
    
    @capture_metadata("integrated_model", "data_loading", "data_loaded")
    def load_data(self, path: str):
        # Automatic metadata capture
        return self._load_data_impl(path)
    
    def train(self, X, y):
        with MetadataCapture("integrated_model", "training", "full_training") as capture:
            capture.add_metadata("algorithm", "CustomAlgorithm")
            
            # Training code with detailed logging
            model = self._train_with_monitoring(X, y, capture)
            
            # Compliance assessment
            self._assess_compliance(X, y, model)
            
            return model
    
    def _assess_compliance(self, X, y, model):
        # Run compliance checks
        bias_metrics = self._calculate_bias(model, X, y)
        
        # Track EEOC compliance
        self.compliance_tracker.track_eeoc_compliance(
            bias_assessment=bias_metrics,
            fairness_metrics=self._calculate_fairness(model, X, y),
            protected_classes=["race", "gender", "age"]
        )
```

## Performance Considerations

### Optimization Tips

1. **Batch Operations**: Use batch writes for high-throughput systems
2. **Async Writes**: Enable asynchronous writes for better performance
3. **Selective Capture**: Only capture necessary metadata to reduce overhead
4. **Compression**: Enable compression for large metadata objects
5. **Cleanup**: Regular cleanup of old metadata to maintain performance

### Monitoring

```python
# Enable performance monitoring
config = MetadataConfig()
config.set("enable_metrics", True)
config.set("metrics_endpoint", "http://monitoring-system:8080/metrics")

# Monitor storage performance
storage_metrics = storage.get_performance_metrics()
print(f"Write latency: {storage_metrics['avg_write_time']}ms")
print(f"Storage size: {storage_metrics['total_size_mb']}MB")
```

## Security and Privacy

### Data Protection

1. **Encryption**: Enable encryption for sensitive metadata
2. **Access Control**: Implement role-based access controls
3. **Audit Logging**: Track all metadata access and modifications
4. **Data Minimization**: Only store necessary information
5. **Retention Policies**: Automatic cleanup of old data

### Configuration

```python
# Security configuration
config.set("encrypt_sensitive_data", True)
config.set("encryption_key_path", "/secure/keys/metadata.key")
config.set("audit_all_access", True)
config.set("alert_on_compliance_failure", True)
```

## Troubleshooting

### Common Issues

1. **Storage Path Permissions**: Ensure write permissions to storage directory
2. **Database Locks**: Use connection pooling for SQLite in multi-threaded apps
3. **Large Metadata**: Consider compression or summary metadata for large objects
4. **Configuration Errors**: Validate configuration before deployment

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Validate configuration
config = MetadataConfig("my_config.json")
errors = config.validate()
if errors:
    print("Configuration errors:", errors)

# Test storage connectivity
try:
    storage = MetadataStorage("test_path", "json")
    test_id = storage.save_metadata("test", "test", "test", {"test": True})
    retrieved = storage.get_metadata(test_id)
    assert retrieved is not None
    print("Storage test passed")
except Exception as e:
    print(f"Storage test failed: {e}")
```

## Best Practices

### Model Development

1. **Initialize Early**: Set up metadata tracking at the start of your project
2. **Comprehensive Logging**: Log all significant pipeline events
3. **Consistent Naming**: Use consistent model names and versions
4. **Stage Organization**: Organize metadata by clear pipeline stages
5. **Documentation**: Include descriptive details for all events

### Production Deployment

1. **Performance Testing**: Test metadata overhead before production
2. **Backup Strategy**: Regular backups of metadata storage
3. **Monitoring**: Monitor storage health and performance
4. **Retention Policies**: Configure appropriate data retention
5. **Compliance Automation**: Automate compliance assessments

### Compliance Management

1. **Framework Selection**: Choose relevant compliance frameworks
2. **Regular Assessment**: Schedule periodic compliance reviews
3. **Documentation**: Maintain detailed compliance documentation
4. **Audit Preparation**: Keep metadata ready for regulatory audits
5. **Continuous Monitoring**: Monitor compliance scores over time

## Advanced Features

### Custom Metadata Validators

```python
def validate_training_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Custom validator for training metadata."""
    errors = []
    
    if metadata.get("accuracy", 0) < 0.8:
        errors.append("Accuracy below minimum threshold")
    
    if not metadata.get("cross_validation"):
        errors.append("Cross-validation results missing")
    
    return errors

# Register validator
storage.register_validator("training", validate_training_metadata)
```

### Custom Compliance Frameworks

```python
class CustomComplianceFramework:
    def __init__(self, name: str, requirements: Dict[str, Any]):
        self.name = name
        self.requirements = requirements
    
    def assess_compliance(self, metadata: Dict[str, Any]) -> float:
        # Custom compliance assessment logic
        score = 0.0
        # ... implementation
        return score

# Register custom framework
tracker.register_framework(CustomComplianceFramework("CUSTOM", {...}))
```

### Integration with External Systems

```python
# Send metadata to external monitoring
def send_to_monitoring(metadata_record: Dict[str, Any]):
    import requests
    requests.post("https://monitoring.example.com/api/metadata", 
                  json=metadata_record)

# Register webhook
storage.register_webhook("metadata_saved", send_to_monitoring)
```

## Migration and Backup

### Migrating Between Backends

```python
# Migrate from JSON to SQLite
json_storage = MetadataStorage("old_metadata", "json")
sqlite_storage = MetadataStorage("new_metadata", "sqlite")

# Export from JSON
export_path = json_storage.export_metadata(format="json")

# Import to SQLite
with open(export_path, 'r') as f:
    data = json.load(f)
    
for record in data:
    sqlite_storage.save_metadata(
        record["model_name"],
        record["stage"],
        record["event_type"],
        record["metadata"],
        record.get("model_version"),
        record.get("details")
    )
```

### Backup and Restore

```python
def backup_metadata(storage: MetadataStorage, backup_path: str):
    """Create backup of all metadata."""
    export_path = storage.export_metadata(format="json")
    
    import shutil
    shutil.copy(export_path, backup_path)
    
    return backup_path

def restore_metadata(backup_path: str, storage: MetadataStorage):
    """Restore metadata from backup."""
    with open(backup_path, 'r') as f:
        data = json.load(f)
    
    for record in data:
        storage.save_metadata(
            record["model_name"],
            record["stage"], 
            record["event_type"],
            record["metadata"],
            record.get("model_version"),
            record.get("details")
        )
```

## Support and Resources

### Getting Help

1. **Documentation**: This comprehensive guide
2. **Examples**: See `metadata_demo.py` for working examples
3. **Setup Script**: Use `setup_metadata.py` for quick project setup
4. **Configuration Templates**: Pre-built configurations for common scenarios

### Contributing

The CIAF metadata storage system is designed to be extensible. Contributions are welcome for:

- New storage backends
- Additional compliance frameworks
- Performance optimizations
- Integration examples
- Documentation improvements

---

For the most up-to-date information and examples, see the CIAF framework documentation and example implementations.
