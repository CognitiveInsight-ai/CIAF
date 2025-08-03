#!/usr/bin/env python3
"""
CIAF Metadata Export Tool

Tool to export metadata from CIAF storage in various formats.
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add CIAF to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from ciaf.metadata_storage import MetadataStorage
    from ciaf.metadata_integration import ModelMetadataManager
except ImportError as e:
    print(f"❌ Could not import CIAF modules: {e}")
    sys.exit(1)


class MetadataExporter:
    """Export metadata in various formats."""
    
    def __init__(self, storage_path: str = "ciaf_metadata"):
        """Initialize with metadata storage path."""
        self.storage = MetadataStorage(
            backend="json",
            storage_path=storage_path
        )
    
    def export_to_json(self, output_path: str, model_filter: Optional[str] = None) -> None:
        """Export metadata to JSON format."""
        try:
            metadata = self.storage.get_all_metadata()
            
            if model_filter:
                metadata = [m for m in metadata if model_filter in m.get('model_id', '')]
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_records': len(metadata),
                'metadata': metadata
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Exported {len(metadata)} records to {output_path}")
            
        except Exception as e:
            print(f"❌ Error exporting to JSON: {e}")
    
    def export_to_csv(self, output_path: str, model_filter: Optional[str] = None) -> None:
        """Export metadata to CSV format."""
        try:
            metadata = self.storage.get_all_metadata()
            
            if model_filter:
                metadata = [m for m in metadata if model_filter in m.get('model_id', '')]
            
            if not metadata:
                print("⚠️ No metadata found to export")
                return
            
            # Flatten metadata for CSV
            flattened_data = []
            for record in metadata:
                flattened = self._flatten_dict(record)
                flattened_data.append(flattened)
            
            # Get all unique keys
            all_keys = set()
            for record in flattened_data:
                all_keys.update(record.keys())
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                writer.writeheader()
                writer.writerows(flattened_data)
            
            print(f"✅ Exported {len(metadata)} records to {output_path}")
            
        except Exception as e:
            print(f"❌ Error exporting to CSV: {e}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def list_models(self) -> None:
        """List all models in metadata storage."""
        try:
            metadata = self.storage.get_all_metadata()
            models = set()
            
            for record in metadata:
                model_id = record.get('model_id')
                if model_id:
                    models.add(model_id)
            
            print(f"📊 Found {len(models)} unique models:")
            for model in sorted(models):
                count = len([m for m in metadata if m.get('model_id') == model])
                print(f"  • {model}: {count} records")
                
        except Exception as e:
            print(f"❌ Error listing models: {e}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Export CIAF metadata")
    parser.add_argument(
        "format", 
        choices=["json", "csv", "list"],
        help="Export format or action"
    )
    parser.add_argument(
        "-o", "--output",
        required=False,
        help="Output file path"
    )
    parser.add_argument(
        "-m", "--model",
        help="Filter by model ID (partial match)"
    )
    parser.add_argument(
        "-s", "--storage",
        default="ciaf_metadata",
        help="Metadata storage path (default: ciaf_metadata)"
    )
    
    args = parser.parse_args()
    
    exporter = MetadataExporter(args.storage)
    
    if args.format == "list":
        exporter.list_models()
    elif args.format == "json":
        if not args.output:
            args.output = f"metadata_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        exporter.export_to_json(args.output, args.model)
    elif args.format == "csv":
        if not args.output:
            args.output = f"metadata_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        exporter.export_to_csv(args.output, args.model)


if __name__ == "__main__":
    main()
