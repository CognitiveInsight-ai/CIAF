# CIAF Metadata Viewer System

## Overview

The CIAF Metadata Viewer System provides a comprehensive solution for storing, managing, and visualizing metadata from machine learning pipelines. This system includes:

- **Metadata Storage Engine**: Multi-backend storage (JSON, SQLite, Pickle)
- **Integration Tools**: Easy-to-use decorators and managers for lifecycle tracking
- **Web Dashboard**: Interactive visualization with multiple view modes
- **Compliance Tracking**: Built-in support for GDPR, FDA, EEOC, FCRA, HIPAA, and ISO 13485

## Quick Start

### 1. Start the Web Application

```bash
cd Test/web
python app.py
```

Then open http://localhost:5000 in your browser.

### 2. Access the Metadata Viewer

1. Navigate to the Compliance Dashboard
2. Click the "View Metadata" button
3. Explore different view modes (Cards, Table, JSON)
4. Use filters to find specific metadata entries

### 3. Generate Sample Data

```bash
python metadata_viewer_demo.py
```

This generates 50+ sample metadata entries for demonstration.

## Core Components

### 1. Metadata Storage (`ciaf/metadata_storage.py`)

The core storage engine supporting multiple backends:

```python
from ciaf.metadata_storage import MetadataStorage

# Initialize storage
storage = MetadataStorage(storage_path="metadata", backend="json")

# Save metadata
metadata_id = storage.save_metadata(
    model_name="my_model",
    stage="training",
    event_type="model_trained",
    metadata={
        "accuracy": 0.95,
        "training_time": "2.5 hours",
        "hyperparameters": {"lr": 0.001, "epochs": 100}
    }
)

# Retrieve metadata
trace = storage.get_pipeline_trace("my_model")
```

### 2. Metadata Integration (`ciaf/metadata_integration.py`)

High-level tools for easy metadata capture:

```python
from ciaf.metadata_integration import ModelMetadataManager, ComplianceTracker

# Model lifecycle tracking
manager = ModelMetadataManager("my_model", "v1.0")

# Log training events
training_id = manager.log_training_start({
    "algorithm": "RandomForest",
    "hyperparameters": {"n_estimators": 100}
})

completion_id = manager.log_training_complete({
    "accuracy": 0.89,
    "training_duration": "45 minutes"
})

# Compliance tracking
tracker = ComplianceTracker(manager)

gdpr_id = tracker.track_gdpr_compliance(
    data_protection_measures={"encryption": True, "anonymization": True},
    consent_management={"consent_obtained": True},
    right_to_explanation=True
)
```

### 3. Configuration Management (`ciaf/metadata_config.py`)

Flexible configuration with templates:

```python
from ciaf.metadata_config import MetadataConfig

# Initialize with GDPR template
config = MetadataConfig.from_template("gdpr")

# Or create custom configuration
config = MetadataConfig(
    storage_backend="sqlite",
    encryption_enabled=True,
    compliance_frameworks=["GDPR", "HIPAA"]
)
```

### 4. Web Dashboard (`Test/web/templates/compliance_dashboard.html`)

Interactive web interface with:

- **Multiple View Modes**: Cards, Table, and JSON views
- **Advanced Filtering**: By date range, model, framework, and custom search
- **Export Functionality**: JSON and CSV export options
- **Real-time Updates**: Live data refresh capabilities
- **Responsive Design**: Works on desktop and mobile devices

## API Endpoints

### GET `/api/metadata/list`
Returns a list of all stored metadata entries.

**Response Format:**
```json
[
  {
    "model_id": "job_classifier_v1",
    "timestamp": "2024-01-15T10:30:00Z",
    "compliance_framework": "EEOC",
    "stage": "training",
    "compliance_score": 0.85,
    "metadata": {
      "accuracy": 0.89,
      "dataset_size": 10000
    }
  }
]
```

### GET `/api/metadata/export?format=csv`
Exports metadata in specified format (json, csv).

## Dashboard Features

### View Modes

1. **Cards View**: Visual cards showing key metadata information
2. **Table View**: Sortable table with pagination
3. **JSON View**: Raw JSON data with syntax highlighting

### Filtering Options

- **Date Range**: Today, This Week, This Month, Custom Range
- **Model Filter**: Filter by specific model names
- **Framework Filter**: Filter by compliance framework
- **Search**: Full-text search across all metadata fields

### Export Options

- **JSON Export**: Complete metadata in JSON format
- **CSV Export**: Flattened metadata for spreadsheet analysis
- **Single Entry Export**: Export individual metadata entries

## Storage Backends

### JSON Storage (Default)
- Human-readable format
- Easy to inspect and debug
- Good for small to medium datasets
- File-based storage in organized directories

### SQLite Storage
- Structured queries with SQL
- Better performance for large datasets
- ACID compliance and transactions
- Built-in indexing and optimization

### Pickle Storage
- Native Python object serialization
- Preserves complex data types
- Fastest for Python-to-Python workflows
- Binary format (not human-readable)

## Compliance Framework Support

The system includes built-in support for major compliance frameworks:

### GDPR (General Data Protection Regulation)
- Data protection measures tracking
- Consent management verification
- Right to explanation compliance
- Data retention and deletion logs

### FDA (Food and Drug Administration)
- Medical device validation tracking
- Clinical trial compliance
- Safety and efficacy metrics
- Audit trail requirements

### EEOC (Equal Employment Opportunity Commission)
- Bias assessment and mitigation
- Protected class monitoring
- Fairness metrics calculation
- Adverse impact analysis

### FCRA (Fair Credit Reporting Act)
- Credit decision accuracy tracking
- Consumer rights compliance
- Dispute resolution logs
- Data quality assurance

### HIPAA (Health Insurance Portability and Accountability Act)
- Healthcare data protection
- Privacy safeguards verification
- Security controls tracking
- Breach notification logs

### ISO 13485 (Medical Devices Quality Management)
- Quality management system compliance
- Risk management tracking
- Device lifecycle documentation
- Regulatory submission support

## File Structure

```
CIAF/
├── ciaf/
│   ├── metadata_storage.py      # Core storage engine
│   ├── metadata_integration.py  # Integration tools
│   ├── metadata_config.py       # Configuration management
│   └── ...
├── Test/web/
│   ├── app.py                    # Flask application
│   └── templates/
│       └── compliance_dashboard.html  # Web interface
├── metadata_viewer_demo.py       # Demonstration script
└── METADATA_VIEWER_README.md    # This documentation
```

## Sample Data Generation

The system includes a comprehensive demo script that generates realistic sample data:

- **50+ Metadata Entries**: Across different models and frameworks
- **Multiple Compliance Frameworks**: GDPR, FDA, EEOC, FCRA, HIPAA coverage
- **Realistic Metrics**: Accuracy, bias scores, compliance ratings
- **Time Series Data**: Entries spanning the last 30 days
- **Model Lifecycle Events**: From data collection to production deployment

## Integration Examples

### Basic Integration

```python
from ciaf.metadata_storage import MetadataStorage

# Simple metadata logging
storage = MetadataStorage()
metadata_id = storage.save_metadata(
    model_name="recommendation_engine",
    stage="inference",
    event_type="prediction_made",
    metadata={
        "user_id": "user_123",
        "prediction": "product_xyz",
        "confidence": 0.87,
        "timestamp": "2024-01-15T10:30:00Z"
    }
)
```

### Advanced Integration with Compliance

```python
from ciaf.metadata_integration import ModelMetadataManager, ComplianceTracker

# Initialize for a specific model
manager = ModelMetadataManager("fraud_detection_v2", "2.1.0")

# Track training process
training_id = manager.log_training_start({
    "dataset_size": 100000,
    "algorithm": "XGBoost",
    "cross_validation_folds": 5
})

# Log compliance check
compliance_tracker = ComplianceTracker(manager)
fcra_compliance = compliance_tracker.track_fcra_compliance(
    accuracy_measures={"precision": 0.92, "recall": 0.88},
    adverse_impact_ratio=0.85,
    data_quality_score=0.94
)
```

### Automated Metadata Capture

```python
from ciaf.metadata_integration import capture_metadata

@capture_metadata(
    model_name="sentiment_analyzer",
    stage="inference",
    capture_inputs=True,
    capture_outputs=True
)
def predict_sentiment(text):
    # Your model inference code here
    return {"sentiment": "positive", "confidence": 0.92}

# Metadata is automatically captured on each function call
result = predict_sentiment("I love this product!")
```

## Performance and Scalability

### Storage Performance
- **JSON**: Good for < 10,000 entries
- **SQLite**: Handles 100,000+ entries efficiently
- **Pickle**: Fastest serialization for complex objects

### Dashboard Performance
- **Pagination**: Displays 12 items per page by default
- **Lazy Loading**: Only loads visible data
- **Client-side Filtering**: Fast search and filter operations
- **Caching**: Optimized API responses

### Memory Usage
- **Streaming Export**: Large datasets exported in chunks
- **Incremental Loading**: Progressive data loading for large collections
- **Connection Pooling**: Efficient database connections

## Security Features

### Data Protection
- **Encryption at Rest**: Optional encryption for sensitive metadata
- **Access Controls**: User-based permissions (when integrated with auth systems)
- **Audit Logging**: Complete audit trail of metadata access and modifications
- **Data Anonymization**: Built-in anonymization utilities

### Privacy Compliance
- **Data Minimization**: Only essential metadata is stored
- **Retention Policies**: Automatic cleanup of old metadata
- **Consent Tracking**: User consent management for data processing
- **Right to Deletion**: Support for data deletion requests

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   Error: cannot import name 'MetadataStorage'
   Solution: Ensure CIAF is in your Python path
   ```

2. **Database Connection Issues**
   ```
   Error: database is locked
   Solution: Check for concurrent access to SQLite database
   ```

3. **Memory Issues with Large Datasets**
   ```
   Error: MemoryError during export
   Solution: Use pagination or switch to SQLite backend
   ```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from ciaf.metadata_storage import MetadataStorage
storage = MetadataStorage(backend="sqlite")  # Enable detailed logging
```

## Future Enhancements

### Planned Features
- **Real-time Monitoring**: Live dashboard updates
- **Advanced Analytics**: Statistical analysis of metadata trends
- **Integration APIs**: REST and GraphQL APIs for external systems
- **Notification System**: Alerts for compliance violations
- **Role-based Access**: Multi-user support with permissions
- **Advanced Visualizations**: Charts and graphs for metadata trends

### Extension Points
- **Custom Storage Backends**: Plugin architecture for new storage types
- **Custom Compliance Frameworks**: Framework-specific plugins
- **Custom Visualizations**: Dashboard widget system
- **External Integrations**: Hooks for third-party systems

## Getting Help

### Documentation
- **API Reference**: Generated from docstrings
- **Examples**: See `metadata_viewer_demo.py` for comprehensive examples
- **Web Interface Help**: Built-in tooltips and help text

### Support
- **GitHub Issues**: Report bugs and feature requests
- **Community Forum**: Ask questions and share experiences
- **Documentation Wiki**: Community-maintained documentation

---

## Conclusion

The CIAF Metadata Viewer System provides a comprehensive solution for ML metadata management with:

✅ **Easy Integration**: Simple APIs and decorators for quick adoption  
✅ **Multiple Storage Options**: Choose the best backend for your needs  
✅ **Compliance Ready**: Built-in support for major regulatory frameworks  
✅ **User-Friendly Interface**: Intuitive web dashboard with multiple view modes  
✅ **Scalable Architecture**: Handles small prototypes to production systems  
✅ **Export Capabilities**: Multiple formats for data analysis and reporting  

Start with the demo script to explore the system, then integrate it into your ML pipelines for comprehensive metadata management and compliance tracking.
