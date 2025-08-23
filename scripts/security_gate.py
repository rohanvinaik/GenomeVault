#!/usr/bin/env python3
"""
Security gate for GenomeVault CI/CD pipeline.

Evaluates security scan results and determines if deployment should proceed.
Implements security policies and provides detailed reporting.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SecurityIssue:
    """Security issue representation."""

    tool: str
    severity: str
    category: str
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None


@dataclass
class SecurityGateConfig:
    """Security gate configuration."""

    # Severity thresholds
    max_critical_issues: int = 0
    max_high_issues: int = 3
    max_medium_issues: int = 10
    max_low_issues: int = 50

    # CVE thresholds
    max_critical_cves: int = 0
    max_high_cves: int = 2
    max_medium_cves: int = 5

    # PHI scanning
    max_phi_critical: int = 0
    max_phi_high: int = 0
    max_phi_medium: int = 2

    # SAST thresholds
    max_sast_critical: int = 0
    max_sast_high: int = 5

    # Container security
    max_container_critical: int = 0
    max_container_high: int = 3

    # License compliance
    blocked_licenses: Set[str] = field(
        default_factory=lambda: {"GPL-2.0", "GPL-3.0", "AGPL-3.0", "LGPL-2.1", "LGPL-3.0"}
    )

    # Required security practices
    require_signed_images: bool = True
    require_sbom: bool = True
    require_vulnerability_scan: bool = True
    require_phi_scan_clean: bool = True


class SecurityGateEvaluator:
    """Evaluates security scan results against policies."""

    def __init__(self, config: SecurityGateConfig):
        self.config = config
        self.issues: List[SecurityIssue] = []
        self.violations: List[str] = []

    def evaluate(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all security scan results."""
        logger.info("Evaluating security gate...")

        # Process different scan types
        if "bandit" in scan_results:
            self._process_bandit_results(scan_results["bandit"])

        if "semgrep" in scan_results:
            self._process_semgrep_results(scan_results["semgrep"])

        if "safety" in scan_results:
            self._process_safety_results(scan_results["safety"])

        if "grype" in scan_results:
            self._process_grype_results(scan_results["grype"])

        if "trivy" in scan_results:
            self._process_trivy_results(scan_results["trivy"])

        if "phi_scan" in scan_results:
            self._process_phi_results(scan_results["phi_scan"])

        # Evaluate against thresholds
        gate_result = self._evaluate_gate()

        return gate_result

    def _process_bandit_results(self, bandit_data: Dict[str, Any]) -> None:
        """Process Bandit SAST results."""
        logger.info("Processing Bandit results...")

        if isinstance(bandit_data, str):
            # Load from file path
            with open(bandit_data, "r") as f:
                bandit_data = json.load(f)

        for result in bandit_data.get("results", []):
            severity_map = {"HIGH": "critical", "MEDIUM": "high", "LOW": "medium"}

            severity = severity_map.get(result.get("issue_severity", "LOW"), "low")

            self.issues.append(
                SecurityIssue(
                    tool="bandit",
                    severity=severity,
                    category="sast",
                    title=result.get("test_name", "Unknown"),
                    description=result.get("issue_text", ""),
                    file_path=result.get("filename"),
                    line_number=result.get("line_number"),
                )
            )

    def _process_semgrep_results(self, semgrep_data: Dict[str, Any]) -> None:
        """Process Semgrep SAST results."""
        logger.info("Processing Semgrep results...")

        if isinstance(semgrep_data, str):
            with open(semgrep_data, "r") as f:
                semgrep_data = json.load(f)

        for result in semgrep_data.get("results", []):
            severity_map = {"ERROR": "critical", "WARNING": "high", "INFO": "medium"}

            severity = severity_map.get(result.get("extra", {}).get("severity", "INFO"), "low")

            self.issues.append(
                SecurityIssue(
                    tool="semgrep",
                    severity=severity,
                    category="sast",
                    title=result.get("check_id", "Unknown"),
                    description=result.get("extra", {}).get("message", ""),
                    file_path=result.get("path"),
                    line_number=result.get("start", {}).get("line"),
                )
            )

    def _process_safety_results(self, safety_data: Dict[str, Any]) -> None:
        """Process Safety dependency scan results."""
        logger.info("Processing Safety results...")

        if isinstance(safety_data, str):
            with open(safety_data, "r") as f:
                safety_data = json.load(f)

        for vuln in safety_data.get("vulnerabilities", []):
            # Map CVE severity to our severity levels
            severity = self._map_cve_severity(vuln.get("vulnerability", {}))

            self.issues.append(
                SecurityIssue(
                    tool="safety",
                    severity=severity,
                    category="dependency",
                    title=f"Vulnerable dependency: {vuln.get('package_name', 'Unknown')}",
                    description=vuln.get("vulnerability", {}).get("description", ""),
                    cve_id=vuln.get("vulnerability", {}).get("id"),
                    cvss_score=vuln.get("vulnerability", {}).get("cvss"),
                )
            )

    def _process_grype_results(self, grype_data: Dict[str, Any]) -> None:
        """Process Grype container scan results."""
        logger.info("Processing Grype results...")

        if isinstance(grype_data, str):
            with open(grype_data, "r") as f:
                grype_data = json.load(f)

        for match in grype_data.get("matches", []):
            vuln = match.get("vulnerability", {})
            severity_map = {
                "Critical": "critical",
                "High": "high",
                "Medium": "medium",
                "Low": "low",
            }

            severity = severity_map.get(vuln.get("severity", "Low"), "low")

            artifact = match.get("artifact", {})

            self.issues.append(
                SecurityIssue(
                    tool="grype",
                    severity=severity,
                    category="container",
                    title=f"Container vulnerability: {vuln.get('id', 'Unknown')}",
                    description=vuln.get("description", ""),
                    file_path=f"{artifact.get('name', 'unknown')}@{artifact.get('version', 'unknown')}",
                    cve_id=vuln.get("id"),
                    cvss_score=self._extract_cvss_score(vuln),
                )
            )

    def _process_trivy_results(self, trivy_data: Dict[str, Any]) -> None:
        """Process Trivy scan results."""
        logger.info("Processing Trivy results...")

        # Trivy SARIF format processing would go here
        # This is a simplified version
        pass

    def _process_phi_results(self, phi_data: Dict[str, Any]) -> None:
        """Process PHI scan results."""
        logger.info("Processing PHI scan results...")

        if isinstance(phi_data, str):
            with open(phi_data, "r") as f:
                phi_data = json.load(f)

        for detection in phi_data.get("detections", []):
            self.issues.append(
                SecurityIssue(
                    tool="phi_scanner",
                    severity=detection.get("severity", "medium"),
                    category="phi_leakage",
                    title=f"PHI detected: {detection.get('type', 'Unknown')}",
                    description=f"Potential PHI leakage: {detection.get('text', '')}",
                    file_path=detection.get("file"),
                    line_number=detection.get("line"),
                )
            )

    def _map_cve_severity(self, vuln_data: Dict[str, Any]) -> str:
        """Map CVE data to severity levels."""
        cvss_score = vuln_data.get("cvss")

        if isinstance(cvss_score, (int, float)):
            if cvss_score >= 9.0:
                return "critical"
            elif cvss_score >= 7.0:
                return "high"
            elif cvss_score >= 4.0:
                return "medium"
            else:
                return "low"

        # Fallback to text severity
        severity_text = vuln_data.get("severity", "").upper()
        severity_map = {"CRITICAL": "critical", "HIGH": "high", "MEDIUM": "medium", "LOW": "low"}

        return severity_map.get(severity_text, "medium")

    def _extract_cvss_score(self, vuln_data: Dict[str, Any]) -> Optional[float]:
        """Extract CVSS score from vulnerability data."""
        # Check various possible locations for CVSS score
        cvss_sources = [
            vuln_data.get("cvss"),
            vuln_data.get("cvssScore"),
            vuln_data.get("severity_score"),
        ]

        for score in cvss_sources:
            if isinstance(score, (int, float)):
                return float(score)

        return None

    def _evaluate_gate(self) -> Dict[str, Any]:
        """Evaluate security gate against thresholds."""
        logger.info("Evaluating security gate thresholds...")

        # Count issues by severity and category
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        category_counts = {}

        for issue in self.issues:
            severity_counts[issue.severity] += 1

            if issue.category not in category_counts:
                category_counts[issue.category] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            category_counts[issue.category][issue.severity] += 1

        # Evaluate thresholds
        passed = True
        threshold_violations = []

        # Overall severity thresholds
        if severity_counts["critical"] > self.config.max_critical_issues:
            passed = False
            threshold_violations.append(
                f"Critical issues: {severity_counts['critical']} > {self.config.max_critical_issues}"
            )

        if severity_counts["high"] > self.config.max_high_issues:
            passed = False
            threshold_violations.append(
                f"High issues: {severity_counts['high']} > {self.config.max_high_issues}"
            )

        if severity_counts["medium"] > self.config.max_medium_issues:
            passed = False
            threshold_violations.append(
                f"Medium issues: {severity_counts['medium']} > {self.config.max_medium_issues}"
            )

        # Category-specific thresholds
        sast_counts = category_counts.get("sast", {"critical": 0, "high": 0})
        if sast_counts["critical"] > self.config.max_sast_critical:
            passed = False
            threshold_violations.append(
                f"SAST critical: {sast_counts['critical']} > {self.config.max_sast_critical}"
            )

        if sast_counts["high"] > self.config.max_sast_high:
            passed = False
            threshold_violations.append(
                f"SAST high: {sast_counts['high']} > {self.config.max_sast_high}"
            )

        # PHI leakage - zero tolerance
        phi_counts = category_counts.get("phi_leakage", {"critical": 0, "high": 0, "medium": 0})
        if phi_counts["critical"] > self.config.max_phi_critical:
            passed = False
            threshold_violations.append(
                f"PHI critical: {phi_counts['critical']} > {self.config.max_phi_critical}"
            )

        if phi_counts["high"] > self.config.max_phi_high:
            passed = False
            threshold_violations.append(
                f"PHI high: {phi_counts['high']} > {self.config.max_phi_high}"
            )

        if phi_counts["medium"] > self.config.max_phi_medium:
            passed = False
            threshold_violations.append(
                f"PHI medium: {phi_counts['medium']} > {self.config.max_phi_medium}"
            )

        # Container security
        container_counts = category_counts.get("container", {"critical": 0, "high": 0})
        if container_counts["critical"] > self.config.max_container_critical:
            passed = False
            threshold_violations.append(
                f"Container critical: {container_counts['critical']} > {self.config.max_container_critical}"
            )

        if container_counts["high"] > self.config.max_container_high:
            passed = False
            threshold_violations.append(
                f"Container high: {container_counts['high']} > {self.config.max_container_high}"
            )

        # Generate report
        result = {
            "passed": passed,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_issues": len(self.issues),
                "severity_breakdown": severity_counts,
                "category_breakdown": category_counts,
                "threshold_violations": threshold_violations,
            },
            "issues": [
                {
                    "tool": issue.tool,
                    "severity": issue.severity,
                    "category": issue.category,
                    "title": issue.title,
                    "description": issue.description,
                    "file": issue.file_path,
                    "line": issue.line_number,
                    "cve_id": issue.cve_id,
                    "cvss_score": issue.cvss_score,
                }
                for issue in self.issues
            ],
            "recommendations": self._generate_recommendations(),
        }

        return result

    def _generate_recommendations(self) -> List[str]:
        """Generate remediation recommendations."""
        recommendations = []

        # Count by category
        category_counts = {}
        for issue in self.issues:
            if issue.category not in category_counts:
                category_counts[issue.category] = 0
            category_counts[issue.category] += 1

        if category_counts.get("sast", 0) > 0:
            recommendations.append("Review and fix SAST findings in source code")

        if category_counts.get("dependency", 0) > 0:
            recommendations.append("Update vulnerable dependencies to latest versions")

        if category_counts.get("container", 0) > 0:
            recommendations.append("Update base images and container dependencies")

        if category_counts.get("phi_leakage", 0) > 0:
            recommendations.append("CRITICAL: Remove PHI from source code immediately")

        # CVE-specific recommendations
        critical_cves = [
            issue for issue in self.issues if issue.severity == "critical" and issue.cve_id
        ]
        if critical_cves:
            recommendations.append(f"Address {len(critical_cves)} critical CVEs immediately")

        return recommendations


def load_scan_results(input_path: str) -> Dict[str, Any]:
    """Load scan results from directory or files."""
    input_path = Path(input_path)
    results = {}

    if input_path.is_dir():
        # Load results from directory
        result_files = {
            "bandit": ["bandit-results.json", "bandit-report.json"],
            "semgrep": ["semgrep-results.json"],
            "safety": ["safety-results.json", "safety-report.json"],
            "grype": ["grype-results.json"],
            "trivy": ["trivy-results.sarif", "trivy-results.json"],
            "phi_scan": ["phi-scan-results.json"],
        }

        for tool, possible_files in result_files.items():
            for filename in possible_files:
                file_path = input_path / filename
                if file_path.exists():
                    results[tool] = str(file_path)
                    logger.info(f"Found {tool} results: {file_path}")
                    break

    else:
        # Single file - determine type by name
        if "bandit" in input_path.name:
            results["bandit"] = str(input_path)
        elif "semgrep" in input_path.name:
            results["semgrep"] = str(input_path)
        elif "safety" in input_path.name:
            results["safety"] = str(input_path)
        elif "grype" in input_path.name:
            results["grype"] = str(input_path)
        elif "phi" in input_path.name:
            results["phi_scan"] = str(input_path)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Security gate evaluation")
    parser.add_argument("--input", required=True, help="Path to scan results directory or file")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument(
        "--threshold",
        choices=["strict", "moderate", "permissive"],
        default="moderate",
        help="Security threshold level",
    )
    parser.add_argument("--bandit", help="Bandit results file")
    parser.add_argument("--semgrep", help="Semgrep results file")
    parser.add_argument("--safety", help="Safety results file")
    parser.add_argument("--grype", help="Grype results file")
    parser.add_argument(
        "--phi-threshold",
        choices=["zero", "low", "medium"],
        default="zero",
        help="PHI tolerance level",
    )

    args = parser.parse_args()

    # Configure security gate based on threshold
    if args.threshold == "strict":
        config = SecurityGateConfig(
            max_critical_issues=0,
            max_high_issues=1,
            max_medium_issues=3,
            max_sast_critical=0,
            max_sast_high=1,
            max_container_critical=0,
            max_container_high=1,
        )
    elif args.threshold == "moderate":
        config = SecurityGateConfig(
            max_critical_issues=0,
            max_high_issues=3,
            max_medium_issues=10,
            max_sast_critical=0,
            max_sast_high=5,
            max_container_critical=0,
            max_container_high=3,
        )
    else:  # permissive
        config = SecurityGateConfig(
            max_critical_issues=1,
            max_high_issues=10,
            max_medium_issues=25,
            max_sast_critical=1,
            max_sast_high=10,
            max_container_critical=1,
            max_container_high=5,
        )

    # Adjust PHI threshold
    if args.phi_threshold == "zero":
        config.max_phi_critical = 0
        config.max_phi_high = 0
        config.max_phi_medium = 0
    elif args.phi_threshold == "low":
        config.max_phi_critical = 0
        config.max_phi_high = 0
        config.max_phi_medium = 2
    else:  # medium
        config.max_phi_critical = 0
        config.max_phi_high = 1
        config.max_phi_medium = 5

    # Load scan results
    scan_results = load_scan_results(args.input)

    # Override with specific file arguments
    if args.bandit:
        scan_results["bandit"] = args.bandit
    if args.semgrep:
        scan_results["semgrep"] = args.semgrep
    if args.safety:
        scan_results["safety"] = args.safety
    if args.grype:
        scan_results["grype"] = args.grype

    if not scan_results:
        logger.error("No scan results found")
        sys.exit(1)

    logger.info(f"Loaded results from: {list(scan_results.keys())}")

    # Evaluate security gate
    evaluator = SecurityGateEvaluator(config)
    result = evaluator.evaluate(scan_results)

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results written to {args.output}")
    else:
        print(json.dumps(result, indent=2))

    # Print summary
    print("\n" + "=" * 60)
    print("SECURITY GATE EVALUATION")
    print("=" * 60)

    status = "✅ PASSED" if result["passed"] else "❌ FAILED"
    print(f"Status: {status}")
    print(f"Total Issues: {result['summary']['total_issues']}")
    print(f"Critical: {result['summary']['severity_breakdown']['critical']}")
    print(f"High: {result['summary']['severity_breakdown']['high']}")
    print(f"Medium: {result['summary']['severity_breakdown']['medium']}")
    print(f"Low: {result['summary']['severity_breakdown']['low']}")

    if result["summary"]["threshold_violations"]:
        print("\nThreshold Violations:")
        for violation in result["summary"]["threshold_violations"]:
            print(f"  ❌ {violation}")

    if result["recommendations"]:
        print("\nRecommendations:")
        for rec in result["recommendations"]:
            print(f"  💡 {rec}")

    # Exit with appropriate code
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
