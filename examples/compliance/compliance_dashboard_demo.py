#!/usr/bin/env python3
"""
CIAF Regulatory Compliance Dashboard Demo

This script demonstrates how to use the CIAF compliance dashboard
and generate compliance reports for regulatory requirements.
"""

import os
import sys
import webbrowser
from pathlib import Path

# Add CIAF to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from tools.compliance.generate_compliance_report import (
        CIAFComplianceReportGenerator,
    )

    COMPLIANCE_TOOLS_AVAILABLE = True
except ImportError:
    COMPLIANCE_TOOLS_AVAILABLE = False


def launch_compliance_dashboard():
    """Launch the interactive compliance dashboard in the browser."""
    dashboard_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "compliance"
        / "regulatory_mapping_dashboard.html"
    )

    if dashboard_path.exists():
        print("🌐 Launching CIAF Regulatory Compliance Dashboard...")
        print(f"📍 Dashboard location: {dashboard_path}")

        # Convert to file URL for browser
        file_url = f"file:///{dashboard_path.absolute().as_posix()}"

        try:
            webbrowser.open(file_url)
            print("✅ Dashboard opened in your default browser!")
            print("\n📋 Dashboard Features:")
            print("   • Interactive overview of all regulatory frameworks")
            print("   • Detailed requirement mappings with CIAF capabilities")
            print("   • Real-time compliance coverage metrics")
            print("   • Implementation guidance and best practices")
            print("   • Searchable requirements database")

        except Exception as e:
            print(f"❌ Could not open browser automatically: {e}")
            print(f"🔗 Please open this URL manually: {file_url}")
    else:
        print(f"❌ Dashboard not found at: {dashboard_path}")
        print(
            "💡 Please ensure the dashboard file exists in the examples/compliance/ directory"
        )


def generate_compliance_reports():
    """Generate detailed compliance reports."""
    if not COMPLIANCE_TOOLS_AVAILABLE:
        print(
            "❌ Compliance tools not available. Please ensure CIAF is properly installed."
        )
        return

    print("📊 Generating CIAF Compliance Reports...")

    try:
        # Initialize report generator
        generator = CIAFComplianceReportGenerator()

        # Generate comprehensive report
        report = generator.generate_comprehensive_report()

        # Save reports with timestamp
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_file = f"ciaf_compliance_report_{timestamp}.json"
        html_file = f"ciaf_compliance_report_{timestamp}.html"

        json_path = generator.save_report(report, json_file)
        html_path = generator.generate_html_report(report, html_file)

        # Display summary
        summary = report["overall_summary"]
        print(f"\n✅ Compliance Assessment Complete!")
        print(f"📈 Overall Compliance Rate: {summary['overall_compliance_rate']}%")
        print(f"🎓 Compliance Grade: {summary['compliance_grade']}")
        print(f"📊 Frameworks Assessed: {summary['frameworks_assessed']}")
        print(f"📋 Total Requirements: {summary['total_requirements']}")

        print(f"\n📄 Reports Generated:")
        print(f"   JSON: {json_path}")
        print(f"   HTML: {html_path}")

        # Ask if user wants to open HTML report
        try:
            open_report = (
                input("\n🌐 Open HTML report in browser? (y/n): ").lower().strip()
            )
            if open_report == "y":
                webbrowser.open(f"file:///{html_path.absolute().as_posix()}")
                print("✅ Report opened in browser!")
        except KeyboardInterrupt:
            print("\n👋 Report generation completed.")

    except Exception as e:
        print(f"❌ Error generating reports: {e}")
        print("💡 This may be due to missing CIAF modules in demo mode.")


def show_framework_summary():
    """Show a summary of supported regulatory frameworks."""
    frameworks = [
        {
            "name": "EU AI Act",
            "description": "European Union Artificial Intelligence Act",
            "coverage": "95%",
            "priority": "High",
            "mandatory": "Yes",
        },
        {
            "name": "NIST AI RMF",
            "description": "NIST AI Risk Management Framework",
            "coverage": "88%",
            "priority": "Medium",
            "mandatory": "Yes",
        },
        {
            "name": "GDPR",
            "description": "General Data Protection Regulation",
            "coverage": "92%",
            "priority": "High",
            "mandatory": "Yes",
        },
        {
            "name": "HIPAA",
            "description": "Health Insurance Portability and Accountability Act",
            "coverage": "85%",
            "priority": "High",
            "mandatory": "Yes",
        },
        {
            "name": "SOX",
            "description": "Sarbanes-Oxley Act",
            "coverage": "78%",
            "priority": "Medium",
            "mandatory": "Yes",
        },
        {
            "name": "ISO 27001",
            "description": "Information Security Management Systems",
            "coverage": "82%",
            "priority": "Medium",
            "mandatory": "No",
        },
    ]

    print("\n🛡️ CIAF Regulatory Framework Support")
    print("=" * 60)

    for framework in frameworks:
        print(f"\n📋 {framework['name']}")
        print(f"   Description: {framework['description']}")
        print(f"   CIAF Coverage: {framework['coverage']}")
        print(f"   Priority: {framework['priority']}")
        print(f"   Mandatory: {framework['mandatory']}")


def show_ciaf_capabilities():
    """Show CIAF capabilities and their regulatory mappings."""
    capabilities = [
        {
            "name": "Cryptographic Integrity",
            "description": "End-to-end cryptographic verification of data and model integrity",
            "frameworks": ["EU AI Act", "GDPR", "HIPAA", "ISO 27001"],
            "requirements": 15,
        },
        {
            "name": "Audit Trails",
            "description": "Comprehensive, tamper-evident audit trails for all system operations",
            "frameworks": ["EU AI Act", "NIST AI RMF", "SOX", "HIPAA"],
            "requirements": 18,
        },
        {
            "name": "Provenance Tracking",
            "description": "Complete lineage tracking from data sources to model outputs",
            "frameworks": ["EU AI Act", "NIST AI RMF", "GDPR"],
            "requirements": 12,
        },
        {
            "name": "Dataset Anchoring",
            "description": "Cryptographic fingerprinting and validation of training datasets",
            "frameworks": ["EU AI Act", "GDPR", "FDA AI/ML"],
            "requirements": 10,
        },
        {
            "name": "Inference Receipts",
            "description": "Verifiable proof of model decisions and reasoning",
            "frameworks": ["EU AI Act", "NIST AI RMF"],
            "requirements": 8,
        },
        {
            "name": "Transparency Reports",
            "description": "Automated generation of compliance and transparency documentation",
            "frameworks": ["EU AI Act", "GDPR", "CCPA"],
            "requirements": 14,
        },
    ]

    print("\n🔧 CIAF Capabilities for Regulatory Compliance")
    print("=" * 60)

    for capability in capabilities:
        print(f"\n⚙️ {capability['name']}")
        print(f"   {capability['description']}")
        print(f"   Addresses {capability['requirements']} requirements")
        print(f"   Frameworks: {', '.join(capability['frameworks'])}")


def main():
    """Main demo function."""
    print("🛡️ CIAF Regulatory Compliance Dashboard Demo")
    print("=" * 50)

    while True:
        print("\n📋 Available Options:")
        print("1. 🌐 Launch Interactive Compliance Dashboard")
        print("2. 📊 Generate Compliance Reports")
        print("3. 📋 Show Framework Summary")
        print("4. 🔧 Show CIAF Capabilities")
        print("5. 🚪 Exit")

        try:
            choice = input("\nSelect an option (1-5): ").strip()

            if choice == "1":
                launch_compliance_dashboard()
            elif choice == "2":
                generate_compliance_reports()
            elif choice == "3":
                show_framework_summary()
            elif choice == "4":
                show_ciaf_capabilities()
            elif choice == "5":
                print("\n👋 Thank you for using CIAF Compliance Tools!")
                break
            else:
                print("❌ Invalid option. Please select 1-5.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()
