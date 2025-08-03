#!/usr/bin/env python3
"""
CIAF Regulatory Compliance Report Generator

This script generates detailed compliance reports mapping CIAF capabilities
to specific regulatory requirements with evidence and verification details.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Add CIAF to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ciaf.compliance.regulatory_mapping import (
        RegulatoryMapper, 
        ComplianceFramework, 
        ComplianceRequirement
    )
    from ciaf.compliance.audit_trails import AuditTrailManager
    from ciaf.core.crypto import CryptographicHash
except ImportError as e:
    print(f"Warning: CIAF modules not fully available: {e}")
    print("Running in demonstration mode with mock data.")
    
    # Mock classes for demonstration
    class ComplianceFramework(Enum):
        EU_AI_ACT = "eu_ai_act"
        NIST_AI_RMF = "nist_ai_rmf"
        GDPR = "gdpr"
        HIPAA = "hipaa"
        SOX = "sox"
        ISO_27001 = "iso_27001"
    
    class ComplianceRequirement:
        def __init__(self, requirement_id, framework, title, description, category, mandatory, ciaf_capabilities, implementation_notes, verification_method, documentation_required, risk_level="medium"):
            self.requirement_id = requirement_id
            self.framework = framework
            self.title = title
            self.description = description
            self.category = category
            self.mandatory = mandatory
            self.ciaf_capabilities = ciaf_capabilities
            self.implementation_notes = implementation_notes
            self.verification_method = verification_method
            self.documentation_required = documentation_required
            self.risk_level = risk_level
    
    class RegulatoryMapper:
        def __init__(self):
            self.mock_requirements = self._create_mock_requirements()
        
        def _create_mock_requirements(self):
            return {
                ComplianceFramework.EU_AI_ACT: [
                    ComplianceRequirement(
                        requirement_id="EU_AI_ACT_001",
                        framework=ComplianceFramework.EU_AI_ACT,
                        title="Risk Management System",
                        description="Establish and maintain a comprehensive risk management system for high-risk AI systems",
                        category="Risk Management",
                        mandatory=True,
                        ciaf_capabilities=["audit_trails", "risk_assessment", "provenance_tracking"],
                        implementation_notes="CIAF provides comprehensive audit trails and risk assessment capabilities",
                        verification_method="Document risk management processes and audit trail integrity",
                        documentation_required=["risk_assessment_report", "audit_trail_documentation"],
                        risk_level="high"
                    )
                ]
            }
        
        def get_requirements_by_framework(self, framework):
            return self.mock_requirements.get(framework, [])
    
    class AuditTrailManager:
        pass
    
    class CryptographicHash:
        @staticmethod
        def generate_hash(data):
            import hashlib
            return hashlib.sha256(str(data).encode()).hexdigest()[:16]


class ComplianceStatus(Enum):
    """Compliance status levels."""
    FULLY_COMPLIANT = "fully_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ComplianceEvidence:
    """Evidence supporting compliance with a requirement."""
    requirement_id: str
    evidence_type: str
    description: str
    file_path: Optional[str] = None
    verification_hash: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ComplianceAssessment:
    """Assessment of compliance for a specific requirement."""
    requirement_id: str
    framework: str
    status: ComplianceStatus
    coverage_percentage: float
    evidence: List[ComplianceEvidence]
    gaps: List[str]
    recommendations: List[str]
    last_assessed: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'requirement_id': self.requirement_id,
            'framework': self.framework,
            'status': self.status.value,
            'coverage_percentage': self.coverage_percentage,
            'evidence': [e.to_dict() for e in self.evidence],
            'gaps': self.gaps,
            'recommendations': self.recommendations,
            'last_assessed': self.last_assessed
        }


class CIAFComplianceReportGenerator:
    """Generates comprehensive compliance reports for CIAF implementations."""
    
    def __init__(self, output_dir: str = "compliance_reports"):
        """Initialize the compliance report generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.regulatory_mapper = RegulatoryMapper()
        self.crypto_hash = CryptographicHash()
        self.audit_manager = AuditTrailManager()
        
        # CIAF capability to implementation mapping
        self.ciaf_implementations = {
            'audit_trails': {
                'module': 'ciaf.compliance.audit_trails',
                'description': 'Comprehensive audit trail system with cryptographic integrity',
                'verification_method': 'Hash chain validation and digital signatures'
            },
            'cryptographic_integrity': {
                'module': 'ciaf.core.cryptographic_core',
                'description': 'End-to-end cryptographic verification system',
                'verification_method': 'Cryptographic hash verification and digital signatures'
            },
            'dataset_anchoring': {
                'module': 'ciaf.anchoring.dataset_anchor',
                'description': 'Cryptographic dataset fingerprinting and validation',
                'verification_method': 'Dataset hash verification and provenance validation'
            },
            'provenance_tracking': {
                'module': 'ciaf.provenance.provenance_capsule',
                'description': 'Complete data and model lineage tracking',
                'verification_method': 'Provenance capsule integrity validation'
            },
            'inference_receipts': {
                'module': 'ciaf.inference.inference_receipt',
                'description': 'Verifiable proof of model decisions and reasoning',
                'verification_method': 'Receipt signature validation and content verification'
            },
            'risk_assessment': {
                'module': 'ciaf.compliance.risk_assessment',
                'description': 'Automated risk assessment and monitoring',
                'verification_method': 'Risk score validation and assessment trail review'
            },
            'transparency_reports': {
                'module': 'ciaf.compliance.transparency_reports',
                'description': 'Automated compliance and transparency documentation',
                'verification_method': 'Report generation audit and content validation'
            },
            'compliance_validation': {
                'module': 'ciaf.compliance.validators',
                'description': 'Automated compliance requirement validation',
                'verification_method': 'Validation result audit and evidence review'
            }
        }
    
    def assess_requirement_compliance(self, requirement: ComplianceRequirement) -> ComplianceAssessment:
        """Assess compliance for a specific requirement."""
        evidence = []
        gaps = []
        recommendations = []
        
        # Calculate coverage based on implemented CIAF capabilities
        implemented_capabilities = []
        missing_capabilities = []
        
        for capability in requirement.ciaf_capabilities:
            if capability in self.ciaf_implementations:
                implemented_capabilities.append(capability)
                # Generate evidence for this capability
                impl = self.ciaf_implementations[capability]
                evidence.append(ComplianceEvidence(
                    requirement_id=requirement.requirement_id,
                    evidence_type="Implementation",
                    description=f"CIAF {capability}: {impl['description']}",
                    verification_hash=self.crypto_hash.generate_hash(impl['module']),
                    timestamp=datetime.now().isoformat()
                ))
            else:
                missing_capabilities.append(capability)
                gaps.append(f"Missing implementation for {capability}")
        
        # Calculate coverage percentage
        if requirement.ciaf_capabilities:
            coverage_percentage = (len(implemented_capabilities) / len(requirement.ciaf_capabilities)) * 100
        else:
            coverage_percentage = 0
        
        # Determine compliance status
        if coverage_percentage >= 95:
            status = ComplianceStatus.FULLY_COMPLIANT
        elif coverage_percentage >= 70:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
            recommendations.append("Implement missing CIAF capabilities to achieve full compliance")
        elif coverage_percentage > 0:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
            recommendations.extend([
                "Significant implementation gaps identified",
                "Priority implementation of core CIAF modules recommended"
            ])
        else:
            status = ComplianceStatus.NON_COMPLIANT
            recommendations.append("Complete CIAF implementation required for compliance")
        
        # Add specific recommendations based on requirement category
        if requirement.category == "Risk Management":
            recommendations.append("Ensure risk assessment processes are documented and regularly updated")
        elif requirement.category == "Data Governance":
            recommendations.append("Implement comprehensive data quality monitoring and validation")
        elif requirement.category == "Transparency":
            recommendations.append("Configure automated transparency report generation")
        
        return ComplianceAssessment(
            requirement_id=requirement.requirement_id,
            framework=requirement.framework.value,
            status=status,
            coverage_percentage=coverage_percentage,
            evidence=evidence,
            gaps=gaps,
            recommendations=recommendations,
            last_assessed=datetime.now().isoformat()
        )
    
    def generate_framework_report(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Generate a comprehensive compliance report for a specific framework."""
        requirements = self.regulatory_mapper.get_requirements_by_framework(framework)
        assessments = []
        
        total_requirements = len(requirements)
        fully_compliant = 0
        partially_compliant = 0
        non_compliant = 0
        
        for requirement in requirements:
            assessment = self.assess_requirement_compliance(requirement)
            assessments.append(assessment)
            
            if assessment.status == ComplianceStatus.FULLY_COMPLIANT:
                fully_compliant += 1
            elif assessment.status == ComplianceStatus.PARTIALLY_COMPLIANT:
                partially_compliant += 1
            else:
                non_compliant += 1
        
        # Calculate overall compliance score
        overall_coverage = sum(a.coverage_percentage for a in assessments) / len(assessments) if assessments else 0
        
        report = {
            'framework': framework.value,
            'framework_name': framework.name if hasattr(framework, 'name') else framework.value.replace('_', ' ').title(),
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_requirements': total_requirements,
                'fully_compliant': fully_compliant,
                'partially_compliant': partially_compliant,
                'non_compliant': non_compliant,
                'overall_coverage_percentage': round(overall_coverage, 2),
                'compliance_score': 'A' if overall_coverage >= 90 else 'B' if overall_coverage >= 80 else 'C' if overall_coverage >= 70 else 'D'
            },
            'assessments': [a.to_dict() for a in assessments],
            'recommendations': self._generate_framework_recommendations(framework, assessments),
            'ciaf_capabilities_used': list(set(
                capability 
                for assessment in assessments 
                for evidence in assessment.evidence
                for capability in self.ciaf_implementations.keys()
                if capability in evidence.description
            ))
        }
        
        return report
    
    def _generate_framework_recommendations(self, framework: ComplianceFramework, assessments: List[ComplianceAssessment]) -> List[str]:
        """Generate framework-specific recommendations."""
        recommendations = []
        
        # Identify common gaps
        all_gaps = [gap for assessment in assessments for gap in assessment.gaps]
        gap_counts = {}
        for gap in all_gaps:
            gap_counts[gap] = gap_counts.get(gap, 0) + 1
        
        # Priority recommendations based on most common gaps
        if gap_counts:
            most_common_gap = max(gap_counts.items(), key=lambda x: x[1])
            recommendations.append(f"Priority: Address {most_common_gap[0]} (affects {most_common_gap[1]} requirements)")
        
        # Framework-specific recommendations
        if framework == ComplianceFramework.EU_AI_ACT:
            recommendations.extend([
                "Implement comprehensive risk management documentation",
                "Ensure human oversight mechanisms are properly documented",
                "Configure automated transparency reporting for high-risk AI systems"
            ])
        elif framework == ComplianceFramework.GDPR:
            recommendations.extend([
                "Implement privacy-by-design principles in all data processing",
                "Ensure data subject rights are properly supported",
                "Configure automated data protection impact assessments"
            ])
        elif framework == ComplianceFramework.HIPAA:
            recommendations.extend([
                "Implement comprehensive access controls for protected health information",
                "Ensure audit trails cover all PHI access and modifications",
                "Configure automated breach detection and reporting"
            ])
        
        return recommendations
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive compliance report across all frameworks."""
        frameworks_to_assess = [
            ComplianceFramework.EU_AI_ACT,
            ComplianceFramework.NIST_AI_RMF,
            ComplianceFramework.GDPR,
            ComplianceFramework.HIPAA,
            ComplianceFramework.SOX,
            ComplianceFramework.ISO_27001
        ]
        
        framework_reports = {}
        for framework in frameworks_to_assess:
            try:
                framework_reports[framework.value] = self.generate_framework_report(framework)
            except Exception as e:
                print(f"Warning: Could not generate report for {framework.value}: {e}")
        
        # Calculate overall statistics
        total_requirements = sum(report['summary']['total_requirements'] for report in framework_reports.values())
        total_fully_compliant = sum(report['summary']['fully_compliant'] for report in framework_reports.values())
        overall_compliance_rate = (total_fully_compliant / total_requirements * 100) if total_requirements > 0 else 0
        
        comprehensive_report = {
            'report_type': 'comprehensive_compliance_assessment',
            'generated_at': datetime.now().isoformat(),
            'ciaf_version': '1.0.0',  # This should be retrieved from CIAF version
            'overall_summary': {
                'frameworks_assessed': len(framework_reports),
                'total_requirements': total_requirements,
                'total_fully_compliant': total_fully_compliant,
                'overall_compliance_rate': round(overall_compliance_rate, 2),
                'compliance_grade': 'A' if overall_compliance_rate >= 90 else 'B' if overall_compliance_rate >= 80 else 'C'
            },
            'framework_reports': framework_reports,
            'ciaf_capabilities_summary': {
                capability: {
                    'description': impl['description'],
                    'verification_method': impl['verification_method'],
                    'requirements_addressed': sum(
                        1 for report in framework_reports.values()
                        for assessment in report['assessments']
                        if capability in [cap for evidence in assessment['evidence'] for cap in self.ciaf_implementations.keys() if cap in evidence['description']]
                    )
                }
                for capability, impl in self.ciaf_implementations.items()
            }
        }
        
        return comprehensive_report
    
    def save_report(self, report: Dict[str, Any], filename: str):
        """Save a report to a JSON file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to: {output_path}")
        return output_path
    
    def generate_html_report(self, report: Dict[str, Any], filename: str):
        """Generate an HTML version of the compliance report."""
        html_content = self._generate_html_content(report)
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(html_content)
        print(f"HTML report saved to: {output_path}")
        return output_path
    
    def _generate_html_content(self, report: Dict[str, Any]) -> str:
        """Generate HTML content for the compliance report."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CIAF Compliance Assessment Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .header h1 {{ color: #2c3e50; margin-bottom: 10px; }}
        .summary {{ background: #e8f5e8; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .framework {{ background: #f8f9fa; padding: 20px; margin-bottom: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .compliance-badge {{ display: inline-block; padding: 5px 15px; border-radius: 15px; color: white; font-weight: bold; margin: 5px; }}
        .fully-compliant {{ background: #28a745; }}
        .partially-compliant {{ background: #ffc107; color: #212529; }}
        .non-compliant {{ background: #dc3545; }}
        .requirement {{ background: white; padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 3px solid #6c757d; }}
        .evidence {{ background: #e8f5e8; padding: 10px; margin: 5px 0; border-radius: 4px; font-size: 0.9em; }}
        .recommendations {{ background: #fff3cd; padding: 15px; border-radius: 6px; margin-top: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ CIAF Compliance Assessment Report</h1>
            <p>Generated on: {report.get('generated_at', 'Unknown')}</p>
        </div>
        
        <div class="summary">
            <h2>Overall Summary</h2>
            <div class="grid">
                <div><strong>Frameworks Assessed:</strong> {report['overall_summary']['frameworks_assessed']}</div>
                <div><strong>Total Requirements:</strong> {report['overall_summary']['total_requirements']}</div>
                <div><strong>Compliance Rate:</strong> {report['overall_summary']['overall_compliance_rate']}%</div>
                <div><strong>Compliance Grade:</strong> <span class="compliance-badge fully-compliant">{report['overall_summary']['compliance_grade']}</span></div>
            </div>
        </div>
        
        <h2>Framework-Specific Reports</h2>
        {''.join(self._generate_framework_html(framework_id, framework_report) for framework_id, framework_report in report['framework_reports'].items())}
        
        <div class="recommendations">
            <h2>🎯 CIAF Capabilities Summary</h2>
            {''.join(f"<div><strong>{cap}:</strong> {details['description']} (Addresses {details['requirements_addressed']} requirements)</div>" for cap, details in report['ciaf_capabilities_summary'].items())}
        </div>
    </div>
</body>
</html>
        """
    
    def _generate_framework_html(self, framework_id: str, framework_report: Dict[str, Any]) -> str:
        """Generate HTML for a specific framework section."""
        summary = framework_report['summary']
        return f"""
        <div class="framework">
            <h3>{framework_report['framework_name']}</h3>
            <div class="grid">
                <div>Total Requirements: {summary['total_requirements']}</div>
                <div>Coverage: {summary['overall_coverage_percentage']}%</div>
                <div>
                    <span class="compliance-badge fully-compliant">{summary['fully_compliant']} Fully Compliant</span>
                    <span class="compliance-badge partially-compliant">{summary['partially_compliant']} Partial</span>
                    <span class="compliance-badge non-compliant">{summary['non_compliant']} Non-Compliant</span>
                </div>
            </div>
        </div>
        """


def main():
    """Main execution function."""
    print("🛡️ CIAF Regulatory Compliance Report Generator")
    print("=" * 50)
    
    # Initialize the report generator
    generator = CIAFComplianceReportGenerator()
    
    # Generate comprehensive compliance report
    print("Generating comprehensive compliance assessment...")
    comprehensive_report = generator.generate_comprehensive_report()
    
    # Save reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"ciaf_compliance_report_{timestamp}.json"
    html_file = f"ciaf_compliance_report_{timestamp}.html"
    
    generator.save_report(comprehensive_report, json_file)
    generator.generate_html_report(comprehensive_report, html_file)
    
    # Print summary
    summary = comprehensive_report['overall_summary']
    print(f"\n📊 Assessment Summary:")
    print(f"   Frameworks Assessed: {summary['frameworks_assessed']}")
    print(f"   Total Requirements: {summary['total_requirements']}")
    print(f"   Overall Compliance: {summary['overall_compliance_rate']}%")
    print(f"   Compliance Grade: {summary['compliance_grade']}")
    
    print(f"\n🎯 Top CIAF Capabilities:")
    capabilities = comprehensive_report['ciaf_capabilities_summary']
    sorted_caps = sorted(capabilities.items(), key=lambda x: x[1]['requirements_addressed'], reverse=True)
    for cap, details in sorted_caps[:5]:
        print(f"   {cap}: {details['requirements_addressed']} requirements")
    
    print(f"\n✅ Reports generated successfully!")
    print(f"   📄 JSON Report: {generator.output_dir / json_file}")
    print(f"   🌐 HTML Report: {generator.output_dir / html_file}")


if __name__ == "__main__":
    main()
