#!/usr/bin/env python3
"""
CIAF Metadata Storage Setup Script

This script helps set up the CIAF metadata storage system for your project.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

def setup_metadata_storage(project_name: str, 
                          backend: str = "json",
                          storage_path: str = None,
                          template: str = "production") -> Dict[str, Any]:
    """
    Set up CIAF metadata storage for a project.
    
    Args:
        project_name: Name of your project
        backend: Storage backend ("json", "sqlite", "pickle")
        storage_path: Custom storage path (optional)
        template: Configuration template to use
    
    Returns:
        Setup configuration
    """
    
    # Import CIAF components
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from ciaf.metadata_storage import MetadataStorage
        from ciaf.metadata_config import MetadataConfig, create_config_template
        from ciaf.metadata_integration import ModelMetadataManager
    except ImportError as e:
        print(f"❌ Error importing CIAF: {e}")
        return None
    
    print(f"🚀 Setting up CIAF metadata storage for '{project_name}'")
    print("="*60)
    
    # Determine storage path
    if storage_path is None:
        storage_path = f"{project_name}_metadata"
    
    # Create configuration
    print(f"📋 Creating configuration from '{template}' template...")
    config_file = f"{project_name}_metadata_config.json"
    
    try:
        create_config_template(template, config_file)
        print(f"✅ Configuration created: {config_file}")
    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        return None
    
    # Update configuration with project-specific settings
    config = MetadataConfig(config_file)
    config.set("storage_backend", backend)
    config.set("storage_path", storage_path)
    config.save_to_file(config_file)
    
    # Initialize storage
    print(f"🗄️ Initializing {backend} storage at '{storage_path}'...")
    try:
        storage = MetadataStorage(storage_path, backend)
        print(f"✅ Storage initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing storage: {e}")
        return None
    
    # Create project directory structure
    project_dir = Path(storage_path)
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    subdirs = ["exports", "backups", "reports"]
    for subdir in subdirs:
        (project_dir / subdir).mkdir(exist_ok=True)
    
    print(f"📁 Created directory structure:")
    print(f"   📂 {storage_path}/")
    for subdir in subdirs:
        print(f"   📂 {storage_path}/{subdir}/")
    
    # Create example integration code
    example_code = f'''"""
{project_name} - CIAF Metadata Integration Example

This module shows how to integrate CIAF metadata storage into your models.
"""

import sys
import os

# Add CIAF to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ciaf.metadata_integration import ModelMetadataManager, ComplianceTracker, capture_metadata
from ciaf.metadata_config import load_config_from_file

# Load project configuration
config = load_config_from_file("{config_file}")

class {project_name.replace('_', '').title()}Model:
    """Example model with CIAF metadata integration."""
    
    def __init__(self, model_name: str, version: str = "1.0.0"):
        self.model_name = model_name
        self.version = version
        
        # Initialize metadata management
        self.metadata_manager = ModelMetadataManager(model_name, version)
        self.compliance_tracker = ComplianceTracker(self.metadata_manager)
    
    @capture_metadata("{project_name}", "data_processing", "data_loaded")
    def load_data(self, data_path: str):
        """Load and process data with automatic metadata capture."""
        # Your data loading code here
        pass
    
    def train(self, X, y):
        """Train model with metadata logging."""
        # Log training start
        training_config = {{
            "algorithm": "YourAlgorithm",
            "samples": len(X),
            "features": X.shape[1] if hasattr(X, 'shape') else len(X[0])
        }}
        self.metadata_manager.log_training_start(training_config)
        
        # Your training code here
        # model = train_your_model(X, y)
        
        # Log training completion
        training_results = {{
            "accuracy": 0.95,  # Replace with actual metrics
            "training_time": 120  # Replace with actual time
        }}
        self.metadata_manager.log_training_complete(training_results)
    
    def predict(self, X):
        """Make predictions with metadata logging."""
        # Your prediction code here
        # predictions = model.predict(X)
        
        # Log inference
        inference_info = {{
            "samples_predicted": len(X),
            "model_version": self.version
        }}
        self.metadata_manager.log_inference(inference_info)
        
        # return predictions

# Example usage
if __name__ == "__main__":
    # Create model instance
    model = {project_name.replace('_', '').title()}Model("my_model", "1.0.0")
    
    # Your model usage here
    # model.load_data("data.csv")
    # model.train(X, y)
    # predictions = model.predict(X_test)
    
    # Get pipeline trace
    trace = model.metadata_manager.get_pipeline_trace()
    print("Pipeline trace:", trace)
'''
    
    example_file = f"{project_name}_metadata_example.py"
    with open(example_file, 'w') as f:
        f.write(example_code)
    
    print(f"📝 Created example integration: {example_file}")
    
    # Create README
    readme_content = f'''# {project_name.title()} - CIAF Metadata Storage

This project uses the CIAF (Cognitive Insight AI Framework) metadata storage system for comprehensive AI pipeline tracking and compliance monitoring.

## Configuration

- **Storage Backend**: {backend}
- **Storage Path**: {storage_path}
- **Configuration File**: {config_file}

## Quick Start

1. **Initialize your model with metadata tracking:**
```python
from ciaf.metadata_integration import ModelMetadataManager

manager = ModelMetadataManager("your_model_name", "1.0.0")
```

2. **Log pipeline events:**
```python
# Data ingestion
manager.log_data_ingestion({{
    "rows": 10000,
    "columns": 50,
    "quality_score": 0.95
}})

# Training
manager.log_training_start({{
    "algorithm": "RandomForest",
    "parameters": {{"n_estimators": 100}}
}})

manager.log_training_complete({{
    "accuracy": 0.94,
    "training_time": 300
}})
```

3. **Track compliance:**
```python
from ciaf.metadata_integration import ComplianceTracker

tracker = ComplianceTracker(manager)
tracker.track_gdpr_compliance(
    data_protection_measures={{"encryption": True}},
    consent_management={{"explicit_consent": True}}
)
```

4. **Use decorators for automatic capture:**
```python
from ciaf.metadata_integration import capture_metadata

@capture_metadata("your_model", "preprocessing", "data_cleaning")
def clean_data(data):
    # Your data cleaning code
    return cleaned_data
```

## Directory Structure

```
{storage_path}/
├── exports/     # Exported metadata files
├── backups/     # Backup files
├── reports/     # Compliance reports
└── your_model/  # Model-specific metadata
```

## Metadata Storage Backends

- **JSON**: Human-readable, easy to inspect and version control
- **SQLite**: Structured querying, better performance for large datasets
- **Pickle**: Python-native serialization, handles complex objects

## Compliance Frameworks Supported

- GDPR (General Data Protection Regulation)
- FDA AI/ML Guidelines
- EEOC (Equal Employment Opportunity Commission)
- FCRA (Fair Credit Reporting Act)
- HIPAA (Health Insurance Portability and Accountability Act)
- ISO 13485 (Medical devices quality management)

## Configuration Options

Edit `{config_file}` to customize:

- Retention policies
- Compliance thresholds
- Export formats
- Security settings
- Performance tuning

## Examples

See `{example_file}` for complete integration examples.

## Support

For more information about CIAF metadata storage, see the main documentation.
'''
    
    readme_file = f"{project_name}_README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"📖 Created README: {readme_file}")
    
    # Return setup summary
    setup_info = {
        "project_name": project_name,
        "backend": backend,
        "storage_path": storage_path,
        "config_file": config_file,
        "example_file": example_file,
        "readme_file": readme_file,
        "status": "success"
    }
    
    print("\n✅ Setup completed successfully!")
    print("\n📋 Summary:")
    print(f"   🔸 Project: {project_name}")
    print(f"   🔸 Backend: {backend}")
    print(f"   🔸 Storage: {storage_path}")
    print(f"   🔸 Config: {config_file}")
    
    print("\n🚀 Next Steps:")
    print(f"1. Review the configuration in '{config_file}'")
    print(f"2. Examine the example code in '{example_file}'")
    print(f"3. Read the setup guide in '{readme_file}'")
    print("4. Integrate metadata tracking into your models")
    
    return setup_info

def main():
    """Interactive setup script."""
    print("🔧 CIAF Metadata Storage Setup Wizard")
    print("="*50)
    
    # Get project information
    project_name = input("\n📝 Enter your project name: ").strip()
    if not project_name:
        project_name = "my_ai_project"
        print(f"   Using default: {project_name}")
    
    # Get storage backend
    print("\n🗄️ Choose storage backend:")
    print("   1. JSON (recommended for small-medium projects)")
    print("   2. SQLite (recommended for large projects)")
    print("   3. Pickle (for complex Python objects)")
    
    backend_choice = input("   Enter choice (1-3) [1]: ").strip()
    backend_map = {"1": "json", "2": "sqlite", "3": "pickle"}
    backend = backend_map.get(backend_choice, "json")
    
    # Get template
    print("\n⚙️ Choose configuration template:")
    print("   1. Development (minimal retention, local storage)")
    print("   2. Production (long retention, full compliance)")
    print("   3. Testing (short retention, fast cleanup)")
    print("   4. High Performance (optimized for speed)")
    
    template_choice = input("   Enter choice (1-4) [2]: ").strip()
    template_map = {"1": "development", "2": "production", "3": "testing", "4": "high_performance"}
    template = template_map.get(template_choice, "production")
    
    # Custom storage path
    custom_path = input(f"\n📁 Custom storage path (press Enter for '{project_name}_metadata'): ").strip()
    storage_path = custom_path if custom_path else None
    
    # Run setup
    print("\n" + "="*50)
    result = setup_metadata_storage(project_name, backend, storage_path, template)
    
    if result and result["status"] == "success":
        print("\n🎉 Setup completed successfully!")
    else:
        print("\n❌ Setup failed!")

if __name__ == "__main__":
    main()
