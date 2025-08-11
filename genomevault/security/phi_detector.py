"""
PHI (Protected Health Information) Leakage Detection.

This module provides tools to scan logs, outputs, and code for potential
PHI leakage patterns to ensure HIPAA compliance.

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PHILeakageDetector:
    """Detect potential PHI leakage in logs, outputs, and code."""

    # Common PHI patterns to detect
    PHI_PATTERNS = {
        "ssn": {
            "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
            "description": "Social Security Number",
            "severity": "critical",
        },
        "npi": {
            "pattern": r"\b\d{10}\b",
            "description": "National Provider Identifier",
            "severity": "high",
            "validator": lambda x: len(x) == 10 and x.isdigit(),
        },
        "genomic_coords": {
            "pattern": r"\bchr[0-9XYM]+[:_]\d+\b",
            "description": "Genomic coordinates",
            "severity": "high",
        },
        "rsid": {
            "pattern": r"\brs\d{4,}\b",
            "description": "SNP rsID",
            "severity": "medium",
        },
        "dob": {
            "pattern": r"\b(0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])[/-](19|20)\d{2}\b",
            "description": "Date of Birth",
            "severity": "high",
        },
        "mrn": {
            "pattern": r"\b(MRN|mrn)[:\s]?\d{6,}\b",
            "description": "Medical Record Number",
            "severity": "critical",
        },
        "email_with_name": {
            "pattern": r"\b[A-Za-z]+\.[A-Za-z]+@[A-Za-z]+\.[A-Za-z]+\b",
            "description": "Email with identifiable name",
            "severity": "medium",
        },
        "phone": {
            "pattern": r"\b(?:\+1-)?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "description": "Phone number",
            "severity": "medium",
        },
        "gene_variant": {
            "pattern": r"\b[A-Z]+\d+[A-Z]+\b",
            "description": "Gene variant notation",
            "severity": "low",
            "validator": lambda x: len(x) <= 10,  # Avoid false positives
        },
        "sequencing_id": {
            "pattern": r"\b(SRR|ERR|DRR)\d{6,}\b",
            "description": "Sequencing run identifier",
            "severity": "medium",
        },
    }

    # Additional context patterns that increase severity
    CONTEXT_PATTERNS = {
        "patient": r"\b(patient|subject|participant|individual)\b",
        "diagnosis": r"\b(diagnosis|diagnosed|condition|disease)\b",
        "treatment": r"\b(treatment|medication|drug|therapy)\b",
        "genetic": r"\b(genetic|genomic|variant|mutation)\b",
    }

    def __init__(self, custom_patterns: dict[str, dict[str, Any]] | None = None):
        """
        Initialize PHI detector.

        Args:
            custom_patterns: Additional patterns to detect
        """
        self.patterns = self.PHI_PATTERNS.copy()
        if custom_patterns:
            self.patterns.update(custom_patterns)

        # Compile regex patterns
        self.compiled_patterns = {}
        for name, info in self.patterns.items():
            self.compiled_patterns[name] = re.compile(info["pattern"], re.IGNORECASE)

        # Compile context patterns
        self.compiled_context = {}
        for name, pattern in self.CONTEXT_PATTERNS.items():
            self.compiled_context[name] = re.compile(pattern, re.IGNORECASE)

    def scan_file(self, filepath: str, max_context: int = 50) -> list[dict[str, Any]]:
        """
        Scan a file for potential PHI leakage.

        Args:
            filepath: Path to file to scan
            max_context: Maximum context characters to include

        Returns:
            List of findings
        """
        findings = []
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    line_findings = self._scan_line(line, line_num, str(filepath), max_context)
                    findings.extend(line_findings)

        except Exception:
            logger.exception("Unhandled exception")
            logger.error(f"Error scanning file {filepath:} {e}")
            raise RuntimeError("Unspecified error")

        return findings

    def scan_logs(self, log_file: str) -> list[dict[str, Any]]:
        """
        Scan logs for potential PHI leakage.

        Args:
            log_file: Path to log file

        Returns:
            List of findings with severity assessment
        """
        return self.scan_file(log_file)

    def scan_directory(
        self,
        directory: str,
        extensions: list[str] = None,
        exclude_patterns: list[str] = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Recursively scan directory for PHI leakage.

        Args:
            directory: Directory to scan
            extensions: File extensions to scan (default: common text/log files)
            exclude_patterns: Patterns to exclude from scanning

        Returns:
            Dict mapping filenames to findings
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Default extensions
        if extensions is None:
            extensions = [".log", ".txt", ".json", ".csv", ".py", ".yaml", ".yml"]

        # Default exclusions
        if exclude_patterns is None:
            exclude_patterns = ["__pycache__", ".git", "node_modules", ".env"]

        all_findings = {}

        for filepath in directory.rglob("*"):
            # Skip directories
            if filepath.is_dir():
                continue

            # Skip excluded patterns
            if any(pattern in str(filepath) for pattern in exclude_patterns):
                continue

            # Check extension
            if extensions and filepath.suffix not in extensions:
                continue

            # Scan file
            findings = self.scan_file(str(filepath))
            if findings:
                all_findings[str(filepath)] = findings

        return all_findings

    def _scan_line(
        self, line: str, line_num: int, filename: str, max_context: int
    ) -> list[dict[str, Any]]:
        """Scan a single line for PHI patterns."""
        findings = []

        for pattern_name, regex in self.compiled_patterns.items():
            matches = list(regex.finditer(line))

            for match in matches:
                # Extract match and context
                start, end = match.span()
                context_start = max(0, start - max_context)
                context_end = min(len(line), end + max_context)

                finding = {
                    "pattern": pattern_name,
                    "description": self.patterns[pattern_name]["description"],
                    "severity": self.patterns[pattern_name]["severity"],
                    "file": filename,
                    "line": line_num,
                    "column": start,
                    "match": match.group(),
                    "context": line[context_start:context_end].strip(),
                    "timestamp": datetime.now().isoformat(),
                }

                # Run validator if present
                if "validator" in self.patterns[pattern_name]:
                    validator = self.patterns[pattern_name]["validator"]
                    if not validator(match.group()):
                        continue

                # Check for context that increases severity
                context_severity = self._assess_context_severity(line)
                if context_severity:
                    finding["context_indicators"] = context_severity
                    # Upgrade severity if context suggests PHI
                    if finding["severity"] == "low" and len(context_severity) >= 2:
                        finding["severity"] = "medium"
                    elif finding["severity"] == "medium" and len(context_severity) >= 2:
                        finding["severity"] = "high"

                findings.append(finding)

        return findings

    def _assess_context_severity(self, line: str) -> list[str]:
        """Check for context patterns that suggest PHI."""
        indicators = []

        for name, regex in self.compiled_context.items():
            if regex.search(line):
                indicators.append(name)

        return indicators

    def redact_phi(self, text: str) -> str:
        """
        Redact potential PHI from text.

        Args:
            text: Text to redact

        Returns:
            Redacted text
        """
        redacted = text

        for pattern_name, regex in self.compiled_patterns.items():
            # Replace matches with redacted version
            def replace_match(match):
                """Replace match.

                Args:
                    match: Match.

                Returns:
                    Operation result.
                """
                matched_text = match.group()
                # Keep first and last char for context
                if len(matched_text) > 2:
                    return matched_text[0] + "[REDACTED]" + matched_text[-1]
                else:
                    return "[REDACTED]"

            redacted = regex.sub(replace_match, redacted)

        return redacted

    def generate_report(self, findings: list[dict[str, Any]], output_format: str = "json") -> str:
        """
        Generate a PHI leakage report.

        Args:
            findings: List of PHI findings
            output_format: Format for report (json, html, markdown)

        Returns:
            Formatted report
        """
        if output_format == "json":
            return json.dumps(
                {
                    "scan_timestamp": datetime.now().isoformat(),
                    "total_findings": len(findings),
                    "findings_by_severity": self._group_by_severity(findings),
                    "findings": findings,
                },
                indent=2,
            )

        elif output_format == "markdown":
            report = ["# PHI Leakage Detection Report\n"]
            report.append(f"**Scan Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report.append(f"**Total Findings**: {len(findings)}\n")

            # Group by severity
            by_severity = self._group_by_severity(findings)

            for severity in ["critical", "high", "medium", "low"]:
                if severity in by_severity:
                    report.append(
                        f"\n## {severity.upper()} Severity ({len(by_severity[severity])})\n"
                    )

                    for finding in by_severity[severity][:10]:  # Limit to 10 per severity
                        report.append(f"- **{finding['description']}** in `{finding['file']}`")
                        report.append(f"  - Line {finding['line']}: `{finding['match']}`")
                        report.append(f"  - Context: `...{finding['context']}...`\n")

            return "\n".join(report)

        elif output_format == "html":
            # Generate HTML report
            html = ["<html><head><title>PHI Leakage Report</title>"]
            html.append("<style>")
            html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
            html.append(".critical { color: #d32f2f; }")
            html.append(".high { color: #f57c00; }")
            html.append(".medium { color: #fbc02d; }")
            html.append(".low { color: #388e3c; }")
            html.append("pre { background: #f5f5f5; padding: 10px; }")
            html.append("</style></head><body>")

            html.append("<h1>PHI Leakage Detection Report</h1>")
            html.append(f"<p>Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            html.append(f"<p>Total Findings: {len(findings)}</p>")

            by_severity = self._group_by_severity(findings)

            for severity in ["critical", "high", "medium", "low"]:
                if severity in by_severity:
                    html.append(
                        f'<h2 class="{severity}">{severity.upper()} ({len(by_severity[severity])})</h2>'
                    )
                    html.append("<ul>")

                    for finding in by_severity[severity][:10]:
                        html.append("<li>")
                        html.append(
                            f"<strong>{finding['description']}</strong> in {finding['file']}<br>"
                        )
                        html.append(f"Line {finding['line']}: <code>{finding['match']}</code><br>")
                        html.append(f"<pre>{finding['context']}</pre>")
                        html.append("</li>")

                    html.append("</ul>")

            html.append("</body></html>")
            return "".join(html)

        else:
            raise ValueError(f"Unknown output format: {output_format}")

    def _group_by_severity(self, findings: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Group findings by severity level."""
        grouped = {}

        for finding in findings:
            severity = finding["severity"]
            if severity not in grouped:
                grouped[severity] = []
            grouped[severity].append(finding)

        return grouped

    def quarantine_file(self, filepath: str, findings: list[dict[str, Any]]) -> str:
        """
        Quarantine a file with PHI leakage.

        Args:
            filepath: File to quarantine
            findings: PHI findings for this file

        Returns:
            Path to quarantined file
        """
        filepath = Path(filepath)

        # Create quarantine directory
        quarantine_dir = Path("quarantine") / datetime.now().strftime("%Y%m%d")
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique quarantine name
        file_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()[:8]
        quarantine_name = f"{filepath.stem}_{file_hash}_QUARANTINED{filepath.suffix}"
        quarantine_path = quarantine_dir / quarantine_name

        # Move file
        filepath.rename(quarantine_path)

        # Create metadata file
        metadata = {
            "original_path": str(filepath),
            "quarantine_date": datetime.now().isoformat(),
            "findings_count": len(findings),
            "findings_summary": self._group_by_severity(findings),
        }

        metadata_path = quarantine_path.with_suffix(".json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.warning(f"Quarantined file with PHI: {filepath} -> {quarantine_path}")

        return str(quarantine_path)


class RealTimePHIMonitor:
    """
    Real-time monitoring for PHI leakage in application outputs.
    """

    def __init__(self, detector: PHILeakageDetector | None = None):
        """
        Initialize real-time monitor.

        Args:
            detector: PHI detector instance (creates default if None)
        """
        self.detector = detector or PHILeakageDetector()
        self.monitoring = False
        self.findings_buffer = []
        self.alert_threshold = 5  # Alert after 5 findings

    def check_output(self, text: str, source: str = "unknown") -> dict[str, Any] | None:
        """
        Check text output for PHI in real-time.

        Args:
            text: Text to check
            source: Source of the text (e.g., "api_response", "log_entry")

        Returns:
            Finding if PHI detected, None otherwise
        """
        # Quick scan for PHI
        for pattern_name, regex in self.detector.compiled_patterns.items():
            match = regex.search(text)
            if match:
                finding = {
                    "pattern": pattern_name,
                    "description": self.detector.patterns[pattern_name]["description"],
                    "severity": self.detector.patterns[pattern_name]["severity"],
                    "source": source,
                    "match": match.group(),
                    "timestamp": datetime.now().isoformat(),
                }

                self.findings_buffer.append(finding)

                # Check if we need to alert
                if len(self.findings_buffer) >= self.alert_threshold:
                    self._trigger_alert()

                return finding

        return None

    def _trigger_alert(self):
        """
        Trigger alert for PHI leakage.
        """
        logger.critical(f"PHI LEAKAGE ALERT: {len(self.findings_buffer)} instances detected!")

        # In production, would send alerts to:
        # - Security team
        # - Compliance officer
        # - Incident response system

        # Clear buffer after alert
        self.findings_buffer = []


# Convenience functions
def scan_genomevault_logs(log_dir: str = "./logs") -> dict[str, list[dict[str, Any]]]:
    """
    Scan GenomeVault logs for PHI leakage.

    Args:
        log_dir: Directory containing logs

    Returns:
        Findings grouped by file
    """
    detector = PHILeakageDetector()
    return detector.scan_directory(log_dir, extensions=[".log"])


def redact_phi_from_file(filepath: str, output_path: str | None = None) -> str:
    """
    Redact PHI from a file and save to new location.

    Args:
        filepath: File to redact
        output_path: Where to save redacted file (auto-generated if None)

    Returns:
        Path to redacted file
    """
    detector = PHILeakageDetector()

    with open(filepath) as f:
        content = f.read()

    redacted_content = detector.redact_phi(content)

    if output_path is None:
        path = Path(filepath)
        output_path = path.parent / f"{path.stem}_redacted{path.suffix}"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(redacted_content)

    return str(output_path)


if __name__ == "__main__":
    # Example usage
    detector = PHILeakageDetector()

    # Test with sample text
    test_text = """
    Patient John Doe (SSN: 123-45-6789) was diagnosed with condition at chr1:12345.
    Their email is john.doe@example.com and MRN: 123456.
    Genetic variant rs1234567 was found. Treatment started on 01/15/1990.
    Contact: 555-123-4567
    """

    logger.info("Original text:")
    logger.info(test_text)

    logger.info("\n" + "=" * 50 + "\n")

    # Scan for PHI
    findings = detector._scan_line(test_text, 1, "test.txt", 50)

    logger.info(f"Found {len(findings)} potential PHI instances:\n")
    for finding in findings:
        logger.info(f"- {finding[}'description']: {finding[}'match'] (Severity: {finding[}'severity'])")

    logger.info("\n" + "=" * 50 + "\n")

    # Redact PHI
    redacted = detector.redact_phi(test_text)
    logger.info("Redacted text:")
    logger.info(redacted)

    # Generate report
    logger.info("\n" + "=" * 50 + "\n")
    logger.info("Markdown Report:")
    logger.info(detector.generate_report(findings, "markdown"))
