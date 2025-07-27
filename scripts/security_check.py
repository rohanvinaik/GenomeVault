#!/usr/bin/env python3
"""
Security check script for GenomeVault.
Verifies log redaction and configuration sanity.
"""
import logging
from typing import Dict, List, Optional, Any, Union

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from genomevault.utils.logging import logger


class SecurityChecker:
    """Security configuration and log checker."""

    def __init__(self) -> None:
            """TODO: Add docstring for __init__"""
    self.issues = []
        self.warnings = []

        # PHI patterns to detect
        self.phi_patterns = {
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "npi": re.compile(r"\b\d{10}\b"),
            "genomic_coord": re.compile(r"\bchr\d+:\d+\b"),
            "rsid": re.compile(r"\brs\d+\b"),
            "dob": re.compile(r"\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}\b"),
            "mrn": re.compile(r"\bMRN\d{6,}\b"),
            "email_phi": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.(com|org|edu|gov)\b"),
        }

    def check_logs(self, log_path: Path) -> List[Dict]:
           """TODO: Add docstring for check_logs"""
     """Check logs for PHI leakage."""
        if not log_path.exists():
            logger.warning(f"Log file not found: {log_path}")
            return []

        findings = []

        with open(log_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                for pattern_name, pattern in self.phi_patterns.items():
                    matches = pattern.findall(line)
                    if matches:
                        findings.append(
                            {
                                "file": str(log_path),
                                "line": line_num,
                                "type": pattern_name,
                                "matches": matches,
                                "severity": "HIGH",
                            }
                        )

        return findings

    def check_config_sanity(self, config_path: Path) -> List[Dict]:
           """TODO: Add docstring for check_config_sanity"""
     """Check configuration for security issues."""
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return []

        issues = []

        with open(config_path, "r") as f:
            config = json.load(f)

        # Check PIR configuration
        if "pir" in config:
            pir_config = config["pir"]

            # Minimum servers check
            if pir_config.get("min_servers", 0) < 2:
                issues.append(
                    {
                        "type": "config",
                        "issue": "PIR requires minimum 2 servers",
                        "severity": "CRITICAL",
                    }
                )

            # Honesty probability check
            if pir_config.get("server_honesty_hipaa", 0) < 0.95:
                issues.append(
                    {"type": "config", "issue": "HIPAA server honesty too low", "severity": "HIGH"}
                )

            # Privacy failure threshold
            if pir_config.get("target_failure_probability", 1.0) > 0.001:
                issues.append(
                    {
                        "type": "config",
                        "issue": "Privacy failure probability too high",
                        "severity": "HIGH",
                    }
                )

        # Check encryption settings
        if "encryption" in config:
            enc_config = config["encryption"]

            # Key length check
            if enc_config.get("key_bits", 0) < 256:
                issues.append(
                    {
                        "type": "config",
                        "issue": "Encryption key length < 256 bits",
                        "severity": "CRITICAL",
                    }
                )

        # Check logging settings
        if "logging" in config:
            log_config = config["logging"]

            # PHI redaction check
            if not log_config.get("redact_phi", False):
                issues.append(
                    {"type": "config", "issue": "PHI redaction not enabled", "severity": "CRITICAL"}
                )

        return issues

    def check_hardcoded_secrets(self, src_dir: Path) -> List[Dict]:
           """TODO: Add docstring for check_hardcoded_secrets"""
     """Check source code for hardcoded secrets."""
        secret_patterns = [
            (re.compile(r'api_key\s*=\s*["\'][^"\']+["\']'), "api_key"),
            (re.compile(r'password\s*=\s*["\'][^"\']+["\']'), "password"),
            (re.compile(r'secret\s*=\s*["\'][^"\']+["\']'), "secret"),
            (re.compile(r'private_key\s*=\s*["\'][^"\']+["\']'), "private_key"),
        ]

        findings = []

        for py_file in src_dir.rglob("*.py"):
            with open(py_file, "r") as f:
                content = f.read()

            for pattern, secret_type in secret_patterns:
                matches = pattern.findall(content)
                if matches:
                    findings.append(
                        {
                            "file": str(py_file),
                            "type": secret_type,
                            "severity": "CRITICAL",
                            "count": len(matches),
                        }
                    )

        return findings

    def generate_report(self) -> Dict:
           """TODO: Add docstring for generate_report"""
     """Generate security check report."""
        return {
            "summary": {
                "total_issues": len(self.issues),
                "critical": len([i for i in self.issues if i.get("severity") == "CRITICAL"]),
                "high": len([i for i in self.issues if i.get("severity") == "HIGH"]),
                "warnings": len(self.warnings),
            },
            "issues": self.issues,
            "warnings": self.warnings,
        }

    def run_all_checks(self, project_dir: Path) -> int:
           """TODO: Add docstring for run_all_checks"""
     """Run all security checks."""
        logger.info("Starting security checks...")

        # Check logs
        log_dir = project_dir / "logs"
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                findings = self.check_logs(log_file)
                self.issues.extend(findings)

        # Check configuration
        config_file = project_dir / "config.json"
        if config_file.exists():
            issues = self.check_config_sanity(config_file)
            self.issues.extend(issues)

        # Check source code
        src_dir = project_dir / "genomevault"
        if src_dir.exists():
            findings = self.check_hardcoded_secrets(src_dir)
            self.issues.extend(findings)

        # Generate report
        report = self.generate_report()

        # Print summary
        print("\n=== Security Check Summary ===")
        print(f"Total Issues: {report['summary']['total_issues']}")
        print(f"  Critical: {report['summary']['critical']}")
        print(f"  High: {report['summary']['high']}")
        print(f"  Warnings: {report['summary']['warnings']}")

        # Print critical issues
        critical_issues = [i for i in self.issues if i.get("severity") == "CRITICAL"]
        if critical_issues:
            print("\nCRITICAL ISSUES:")
            for issue in critical_issues:
                print(f"  - {issue}")

        # Return exit code
        return 1 if report["summary"]["critical"] > 0 else 0


def main() -> None:
       """TODO: Add docstring for main"""
     """Main entry point."""
    parser = argparse.ArgumentParser(description="GenomeVault security checker")
    parser.add_argument(
        "--project-dir", type=Path, default=Path.cwd(), help="Project directory to check"
    )
    parser.add_argument("--log-file", type=Path, help="Specific log file to check")
    parser.add_argument("--config-file", type=Path, help="Specific config file to check")

    args = parser.parse_args()

    checker = SecurityChecker()

    if args.log_file:
        # Check specific log file
        findings = checker.check_logs(args.log_file)
        if findings:
            print(f"Found {len(findings)} PHI instances in {args.log_file}")
            for finding in findings:
                print(f"  Line {finding['line']}: {finding['type']} - {finding['matches']}")
        else:
            print("No PHI found in log file")

    elif args.config_file:
        # Check specific config
        issues = checker.check_config_sanity(args.config_file)
        if issues:
            print(f"Found {len(issues)} configuration issues")
            for issue in issues:
                print(f"  [{issue['severity']}] {issue['issue']}")
        else:
            print("Configuration looks secure")

    else:
        # Run all checks
        exit_code = checker.run_all_checks(args.project_dir)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
