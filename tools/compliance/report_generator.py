#!/usr/bin/env python3
"""
CIAF Compliance Report Generator

Tool to generate compliance reports for various regulatory frameworks.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add CIAF to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from ciaf.compliance.regulatory_mapping import RegulatoryMapper, ComplianceFramework
    from ciaf.compliance.reports import ComplianceReportGenerator, ReportType
    from ciaf.metadata_storage import MetadataStorage
except ImportError as e:
    print(f"❌ Could not import CIAF modules: {e}")
    sys.exit(1)


class ComplianceReporter:
    """Generate compliance reports."""
    
    def __init__(self, storage_path: str = "ciaf_metadata"):
        """Initialize with metadata storage path."""
        self.storage = MetadataStorage(
            backend="json",
            storage_path=storage_path
        )
        self.mapper = RegulatoryMapper()
        
    def generate_framework_report(
        self, 
        framework: ComplianceFramework,
        model_id: str,
        output_path: Optional[str] = None
    ) -> None:
        """Generate compliance report for a specific framework and model."""
        try:
            # Get model metadata
            metadata = self.storage.get_model_trace(model_id)
            if not metadata:
                print(f"❌ No metadata found for model: {model_id}")
                return
            
            # Generate compliance mapping
            coverage = self.mapper.get_framework_coverage(framework, metadata)
            
            # Create report
            report = {
                'report_info': {
                    'framework': framework.value,
                    'model_id': model_id,
                    'generated_at': datetime.now().isoformat(),
                    'ciaf_version': '2.1.0'
                },
                'executive_summary': {
                    'overall_coverage': coverage['overall_coverage']['coverage_percentage'],
                    'total_requirements': coverage['overall_coverage']['total_requirements'],
                    'satisfied_requirements': coverage['overall_coverage']['satisfied_requirements'],
                    'automation_level': coverage['overall_coverage'].get('automation_level', 0.0)
                },
                'detailed_coverage': coverage,
                'recommendations': self._generate_recommendations(coverage),
                'audit_trail': metadata
            }
            
            # Output path
            if not output_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"compliance_report_{framework.value}_{model_id}_{timestamp}.json"
            
            # Save report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✅ Generated compliance report: {output_path}")
            print(f"📊 Coverage: {coverage['overall_coverage']['coverage_percentage']:.1f}%")
            print(f"📋 Requirements: {coverage['overall_coverage']['satisfied_requirements']}/{coverage['overall_coverage']['total_requirements']}")
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
    
    def _generate_recommendations(self, coverage: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations based on coverage analysis."""
        recommendations = []
        
        # Check overall coverage
        overall_percentage = coverage['overall_coverage']['coverage_percentage']
        if overall_percentage < 80:
            recommendations.append({
                'priority': 'high',
                'category': 'coverage',
                'title': 'Improve Overall Compliance Coverage',
                'description': f'Current coverage is {overall_percentage:.1f}%. Recommend implementing additional compliance measures to reach 80%+ coverage.'
            })
        
        # Check stage-specific coverage
        for stage_name, stage_data in coverage.get('stage_coverage', {}).items():
            stage_percentage = stage_data.get('coverage_percentage', 0)
            if stage_percentage < 70:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'stage_coverage',
                    'title': f'Improve {stage_name.replace("_", " ").title()} Stage Coverage',
                    'description': f'{stage_name} stage has {stage_percentage:.1f}% coverage. Consider implementing additional controls for this stage.'
                })
        
        # Check automation level
        automation_level = coverage['overall_coverage'].get('automation_level', 0.0)
        if automation_level < 0.7:
            recommendations.append({
                'priority': 'low',
                'category': 'automation',
                'title': 'Increase Automation Level',
                'description': f'Current automation level is {automation_level:.1f}%. Consider automating more compliance checks to reduce manual effort.'
            })
        
        return recommendations
    
    def list_available_frameworks(self) -> None:
        """List all available compliance frameworks."""
        print("📋 Available Compliance Frameworks:")
        for framework in ComplianceFramework:
            print(f"  • {framework.value}: {framework.name}")
    
    def list_models_with_metadata(self) -> None:
        """List all models with available metadata."""
        try:
            metadata = self.storage.get_all_metadata()
            models = {}
            
            for record in metadata:
                model_id = record.get('model_id')
                if model_id:
                    if model_id not in models:
                        models[model_id] = []
                    models[model_id].append(record.get('stage', 'unknown'))
            
            print(f"📊 Found {len(models)} models with metadata:")
            for model_id, stages in models.items():
                unique_stages = set(stages)
                print(f"  • {model_id}: {len(stages)} records, {len(unique_stages)} stages")
                
        except Exception as e:
            print(f"❌ Error listing models: {e}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Generate CIAF compliance reports")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate report command
    generate_parser = subparsers.add_parser('generate', help='Generate compliance report')
    generate_parser.add_argument('framework', help='Compliance framework (e.g., eu_ai_act, nist_ai_rmf)')
    generate_parser.add_argument('model_id', help='Model ID to generate report for')
    generate_parser.add_argument('-o', '--output', help='Output file path')
    generate_parser.add_argument('-s', '--storage', default='ciaf_metadata', help='Metadata storage path')
    
    # List frameworks command
    list_frameworks_parser = subparsers.add_parser('frameworks', help='List available frameworks')
    
    # List models command
    list_models_parser = subparsers.add_parser('models', help='List models with metadata')
    list_models_parser.add_argument('-s', '--storage', default='ciaf_metadata', help='Metadata storage path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'frameworks':
        reporter = ComplianceReporter()
        reporter.list_available_frameworks()
    elif args.command == 'models':
        reporter = ComplianceReporter(args.storage)
        reporter.list_models_with_metadata()
    elif args.command == 'generate':
        try:
            framework = ComplianceFramework(args.framework)
        except ValueError:
            print(f"❌ Invalid framework: {args.framework}")
            print("Use 'frameworks' command to see available options")
            return
        
        reporter = ComplianceReporter(args.storage)
        reporter.generate_framework_report(framework, args.model_id, args.output)


if __name__ == "__main__":
    main()
