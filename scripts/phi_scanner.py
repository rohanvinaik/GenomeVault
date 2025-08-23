#!/usr/bin/env python3
"""
PHI (Protected Health Information) scanner for GenomeVault.

Scans source code, documentation, and configuration files for potential
PHI leakage including:
- Personal identifiers (SSN, MRN, etc.)
- Genomic data patterns
- Clinical information
- Patient data structures
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Set, Tuple, Union

# PHI Detection Patterns
PHI_PATTERNS = {
    # Personal Identifiers
    "ssn": [
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # XXX-XX-XXXX
        re.compile(r"\b\d{9}\b"),  # XXXXXXXXX
    ],
    "medical_record_number": [
        re.compile(r"\bMRN[:\s]*\d+\b", re.IGNORECASE),
        re.compile(r"\bmedical.record.number[:\s]*\d+\b", re.IGNORECASE),
        re.compile(r"\bpatient.id[:\s]*\d+\b", re.IGNORECASE),
    ],
    "phone_number": [
        re.compile(r"\b\d{3}-\d{3}-\d{4}\b"),  # XXX-XXX-XXXX
        re.compile(r"\(\d{3}\)\s*\d{3}-\d{4}"),  # (XXX) XXX-XXXX
    ],
    "email_address": [
        re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
    ],
    # Genomic Data Patterns
    "genomic_coordinates": [
        re.compile(r"\bchr[0-9XYM]+:\d+\b", re.IGNORECASE),  # chr1:12345
        re.compile(r"\b[0-9XYM]+:\d+-\d+\b"),  # 1:12345-67890
    ],
    "rsid": [
        re.compile(r"\brs\d+\b", re.IGNORECASE),  # rs123456
    ],
    "genotype": [
        re.compile(r"\b[0-9]/[0-9]\b"),  # 0/1, 1/1
        re.compile(r"\b[ATCG]/[ATCG]\b"),  # A/T, G/C
    ],
    "dna_sequence": [
        re.compile(r"\b[ATCGN]{20,}\b"),  # Long DNA sequences
    ],
    # Clinical Information
    "diagnosis_codes": [
        re.compile(r"\bICD-?10[:\s]*[A-Z]\d{2}\.?\d*\b", re.IGNORECASE),  # ICD-10
        re.compile(r"\bICD-?9[:\s]*\d{3}\.?\d*\b", re.IGNORECASE),  # ICD-9
    ],
    "medication": [
        re.compile(r"\b\d+mg\b", re.IGNORECASE),
        re.compile(r"\bmg/kg\b", re.IGNORECASE),
    ],
    # Dates (potential birthdates)
    "dates": [
        re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"),  # MM/DD/YYYY
        re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # YYYY-MM-DD
    ],
    # Database/System Identifiers
    "database_ids": [
        re.compile(r"\bpatient_\d+\b", re.IGNORECASE),
        re.compile(r"\buser_\d+\b", re.IGNORECASE),
        re.compile(r"\bsubject_\d+\b", re.IGNORECASE),
    ],
    # Clinical Terms
    "clinical_terms": [
        re.compile(r"\bdiagnos[ie]s\b", re.IGNORECASE),
        re.compile(r"\bsymptom\b", re.IGNORECASE),
        re.compile(r"\btreatment\b", re.IGNORECASE),
        re.compile(r"\bmedication\b", re.IGNORECASE),
        re.compile(r"\ballergy\b", re.IGNORECASE),
    ],
}

# Genomic Data Specific Patterns
GENOMIC_PATTERNS = {
    "vcf_format": [
        re.compile(r"##fileformat=VCFv", re.IGNORECASE),
        re.compile(r"#CHROM\tPOS\tID\tREF\tALT", re.IGNORECASE),
    ],
    "fasta_format": [
        re.compile(r">[A-Za-z0-9_|.-]+"),  # FASTA headers
    ],
    "chromosome_notation": [
        re.compile(r"\bchr([1-9]|1[0-9]|2[0-2]|X|Y|M)\b", re.IGNORECASE),
    ],
    "genomic_ranges": [
        re.compile(r"\b\d+bp\b", re.IGNORECASE),  # base pairs
        re.compile(r"\b\d+kb\b", re.IGNORECASE),  # kilobases
        re.compile(r"\b\d+Mb\b", re.IGNORECASE),  # megabases
    ],
}

# Allowlisted patterns (legitimate use in code)
ALLOWLIST_PATTERNS = [
    re.compile(r"example\.com"),
    re.compile(r"test@example\.org"),
    re.compile(r"xxx-xx-xxxx", re.IGNORECASE),  # Masked examples
    re.compile(r"patient_id.*example", re.IGNORECASE),
    re.compile(r"rs\d+.*example", re.IGNORECASE),
    re.compile(r"chr\d+.*test", re.IGNORECASE),
]

# File extensions to scan
SCANNABLE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".md",
    ".txt",
    ".rst",
    ".yaml",
    ".yml",
    ".json",
    ".xml",
    ".sql",
    ".sh",
    ".bash",
    ".dockerfile",
    ".env",
}

# Directories to skip
SKIP_DIRECTORIES = {
    ".git",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    "venv",
    "env",
    ".venv",
    "dist",
    "build",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    ".coverage",
    ".benchmarks",
}


@dataclass
class PHIDetection:
    """PHI detection result."""

    file_path: str
    line_number: int
    pattern_type: str
    matched_text: str
    context: str
    severity: str = "medium"
    confidence: float = 0.8


@dataclass
class ScanResults:
    """Scan results summary."""

    total_files_scanned: int = 0
    detections: List[PHIDetection] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)

    def add_detection(
        self,
        file_path: str,
        line_number: int,
        pattern_type: str,
        matched_text: str,
        context: str,
        severity: str = "medium",
        confidence: float = 0.8,
    ) -> None:
        """Add PHI detection."""
        self.detections.append(
            PHIDetection(
                file_path=file_path,
                line_number=line_number,
                pattern_type=pattern_type,
                matched_text=matched_text,
                context=context,
                severity=severity,
                confidence=confidence,
            )
        )

    def get_high_confidence_detections(self, threshold: float = 0.9) -> List[PHIDetection]:
        """Get high confidence detections."""
        return [d for d in self.detections if d.confidence >= threshold]

    def get_by_severity(self, severity: str) -> List[PHIDetection]:
        """Get detections by severity."""
        return [d for d in self.detections if d.severity == severity]


class PHIScanner:
    """PHI scanner implementation."""

    def __init__(
        self, strict_mode: bool = False, check_comments: bool = True, check_docstrings: bool = True
    ):
        self.strict_mode = strict_mode
        self.check_comments = check_comments
        self.check_docstrings = check_docstrings
        self.results = ScanResults()

    def scan_path(self, path: Union[str, Path]) -> ScanResults:
        """Scan path for PHI."""
        path = Path(path)

        if path.is_file():
            self._scan_file(path)
        elif path.is_dir():
            self._scan_directory(path)
        else:
            self.results.errors.append(f"Path not found: {path}")

        return self.results

    def _scan_directory(self, directory: Path) -> None:
        """Recursively scan directory."""
        for item in directory.rglob("*"):
            if item.is_file():
                # Skip if in skip directory
                if any(skip_dir in item.parts for skip_dir in SKIP_DIRECTORIES):
                    continue

                # Check if file extension is scannable
                if item.suffix.lower() in SCANNABLE_EXTENSIONS or item.name.startswith(".env"):
                    self._scan_file(item)
                else:
                    self.results.skipped_files.append(str(item))

    def _scan_file(self, file_path: Path) -> None:
        """Scan individual file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            self.results.total_files_scanned += 1

            for line_num, line in enumerate(lines, 1):
                self._scan_line(file_path, line_num, line)

        except Exception as e:
            self.results.errors.append(f"Error scanning {file_path}: {str(e)}")

    def _scan_line(self, file_path: Path, line_num: int, line: str) -> None:
        """Scan individual line for PHI."""
        # Skip certain lines in non-strict mode
        if not self.strict_mode:
            line_lower = line.lower()

            # Skip comments unless enabled
            if not self.check_comments and (
                line.strip().startswith("#") or line.strip().startswith("//")
            ):
                return

            # Skip docstrings unless enabled
            if not self.check_docstrings and ('"""' in line or "'''" in line):
                return

            # Skip obvious test/example lines
            if any(
                keyword in line_lower for keyword in ["example", "test", "sample", "dummy", "fake"]
            ):
                return

        # Check allowlist first
        if self._is_allowlisted(line):
            return

        # Scan for PHI patterns
        self._check_phi_patterns(file_path, line_num, line)
        self._check_genomic_patterns(file_path, line_num, line)

    def _is_allowlisted(self, line: str) -> bool:
        """Check if line matches allowlist patterns."""
        return any(pattern.search(line) for pattern in ALLOWLIST_PATTERNS)

    def _check_phi_patterns(self, file_path: Path, line_num: int, line: str) -> None:
        """Check line for PHI patterns."""
        for pattern_type, patterns in PHI_PATTERNS.items():
            for pattern in patterns:
                matches = pattern.finditer(line)
                for match in matches:
                    severity, confidence = self._assess_phi_risk(pattern_type, match.group(), line)

                    # Only report if meets threshold
                    if confidence >= (0.9 if self.strict_mode else 0.7):
                        self.results.add_detection(
                            file_path=str(file_path),
                            line_number=line_num,
                            pattern_type=pattern_type,
                            matched_text=match.group(),
                            context=line.strip(),
                            severity=severity,
                            confidence=confidence,
                        )

    def _check_genomic_patterns(self, file_path: Path, line_num: int, line: str) -> None:
        """Check line for genomic data patterns."""
        for pattern_type, patterns in GENOMIC_PATTERNS.items():
            for pattern in patterns:
                matches = pattern.finditer(line)
                for match in matches:
                    # Genomic data in code may be legitimate, assess context
                    severity, confidence = self._assess_genomic_risk(
                        pattern_type, match.group(), line, file_path
                    )

                    if confidence >= (0.8 if self.strict_mode else 0.6):
                        self.results.add_detection(
                            file_path=str(file_path),
                            line_number=line_num,
                            pattern_type=f"genomic_{pattern_type}",
                            matched_text=match.group(),
                            context=line.strip(),
                            severity=severity,
                            confidence=confidence,
                        )

    def _assess_phi_risk(self, pattern_type: str, match: str, context: str) -> Tuple[str, float]:
        """Assess PHI risk severity and confidence."""
        context_lower = context.lower()

        # High-risk patterns
        if pattern_type in ["ssn", "medical_record_number"]:
            # Check if it's in a comment or example
            if any(keyword in context_lower for keyword in ["example", "test", "sample", "#"]):
                return "low", 0.3
            return "critical", 0.95

        # Genomic identifiers
        if pattern_type in ["rsid", "genomic_coordinates"]:
            # Legitimate in genomics code, but check context
            if any(keyword in context_lower for keyword in ["patient", "subject", "individual"]):
                return "high", 0.85
            return "medium", 0.6

        # Clinical terms - context dependent
        if pattern_type == "clinical_terms":
            if any(keyword in context_lower for keyword in ["patient", "diagnosis", "treatment"]):
                return "medium", 0.7
            return "low", 0.4

        # Phone/email - context dependent
        if pattern_type in ["phone_number", "email_address"]:
            if any(keyword in context_lower for keyword in ["contact", "patient", "doctor"]):
                return "high", 0.8
            return "medium", 0.6

        return "medium", 0.7

    def _assess_genomic_risk(
        self, pattern_type: str, match: str, context: str, file_path: Path
    ) -> Tuple[str, float]:
        """Assess genomic data risk."""
        context_lower = context.lower()
        file_name = file_path.name.lower()

        # VCF/FASTA format headers are usually legitimate
        if pattern_type in ["vcf_format", "fasta_format"]:
            return "low", 0.3

        # Real genomic coordinates in non-test files
        if pattern_type in ["genomic_coordinates", "chromosome_notation"]:
            # High risk if appears to be real patient data
            if any(
                keyword in context_lower
                for keyword in ["patient", "subject", "individual", "sample"]
            ):
                return "high", 0.9

            # Medium risk in configuration or data files
            if any(ext in file_name for ext in [".json", ".yaml", ".csv", ".txt"]):
                return "medium", 0.7

            # Low risk in code (probably examples/tests)
            return "low", 0.4

        # DNA sequences
        if pattern_type == "dna_sequence":
            # Very long sequences are suspicious
            if len(match) > 100:
                return "high", 0.8
            return "medium", 0.6

        return "medium", 0.6


def generate_report(results: ScanResults, format: str = "text") -> str:
    """Generate scan report."""
    if format == "json":
        return json.dumps(
            {
                "summary": {
                    "total_files_scanned": results.total_files_scanned,
                    "total_detections": len(results.detections),
                    "critical_detections": len(results.get_by_severity("critical")),
                    "high_detections": len(results.get_by_severity("high")),
                    "medium_detections": len(results.get_by_severity("medium")),
                    "low_detections": len(results.get_by_severity("low")),
                    "high_confidence_detections": len(results.get_high_confidence_detections()),
                    "errors": len(results.errors),
                },
                "detections": [
                    {
                        "file": d.file_path,
                        "line": d.line_number,
                        "type": d.pattern_type,
                        "text": d.matched_text,
                        "context": d.context,
                        "severity": d.severity,
                        "confidence": d.confidence,
                    }
                    for d in results.detections
                ],
                "errors": results.errors,
            },
            indent=2,
        )

    # Text format
    report = []
    report.append("=" * 60)
    report.append("PHI SCANNER REPORT")
    report.append("=" * 60)
    report.append(f"Files scanned: {results.total_files_scanned}")
    report.append(f"Total detections: {len(results.detections)}")
    report.append(f"Critical: {len(results.get_by_severity('critical'))}")
    report.append(f"High: {len(results.get_by_severity('high'))}")
    report.append(f"Medium: {len(results.get_by_severity('medium'))}")
    report.append(f"Low: {len(results.get_by_severity('low'))}")
    report.append("")

    if results.detections:
        report.append("DETECTIONS:")
        report.append("-" * 40)

        # Sort by severity and confidence
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_detections = sorted(
            results.detections, key=lambda x: (severity_order.get(x.severity, 4), -x.confidence)
        )

        for detection in sorted_detections:
            severity_icon = {"critical": "🚨", "high": "⚠️", "medium": "⚡", "low": "ℹ️"}.get(
                detection.severity, "❓"
            )

            report.append(
                f"{severity_icon} {detection.severity.upper()} - {detection.pattern_type}"
            )
            report.append(f"   File: {detection.file_path}:{detection.line_number}")
            report.append(f"   Match: {detection.matched_text}")
            report.append(f"   Context: {detection.context[:100]}...")
            report.append(f"   Confidence: {detection.confidence:.2f}")
            report.append("")

    if results.errors:
        report.append("ERRORS:")
        report.append("-" * 40)
        for error in results.errors:
            report.append(f"❌ {error}")
        report.append("")

    return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Scan for PHI leakage in code")
    parser.add_argument("path", help="Path to scan")
    parser.add_argument("--strict", action="store_true", help="Strict scanning mode")
    parser.add_argument(
        "--no-comments", dest="check_comments", action="store_false", help="Skip comments"
    )
    parser.add_argument(
        "--no-docstrings", dest="check_docstrings", action="store_false", help="Skip docstrings"
    )
    parser.add_argument(
        "--output-format", choices=["text", "json"], default="text", help="Output format"
    )
    parser.add_argument("--output-file", help="Output file (default: stdout)")
    parser.add_argument(
        "--threshold",
        choices=["critical", "high", "medium", "low"],
        default="medium",
        help="Minimum severity to report",
    )

    args = parser.parse_args()

    # Initialize scanner
    scanner = PHIScanner(
        strict_mode=args.strict,
        check_comments=args.check_comments,
        check_docstrings=args.check_docstrings,
    )

    # Scan path
    results = scanner.scan_path(args.path)

    # Filter by threshold
    threshold_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    min_level = threshold_levels[args.threshold]

    filtered_detections = [
        d for d in results.detections if threshold_levels.get(d.severity, 0) >= min_level
    ]
    results.detections = filtered_detections

    # Generate report
    report = generate_report(results, args.output_format)

    # Output report
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(report)
        print(f"Report written to {args.output_file}")
    else:
        print(report)

    # Exit with appropriate code
    critical_issues = len(results.get_by_severity("critical"))
    high_issues = len(results.get_by_severity("high"))

    if critical_issues > 0:
        print(f"\n🚨 {critical_issues} critical PHI issues found!")
        sys.exit(2)
    elif high_issues > 0:
        print(f"\n⚠️  {high_issues} high-severity PHI issues found!")
        sys.exit(1)
    elif len(results.detections) > 0:
        print(f"\nℹ️  {len(results.detections)} potential PHI issues found.")
        sys.exit(0)
    else:
        print("\n✅ No PHI issues detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
