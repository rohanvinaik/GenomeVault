"""
Validation utilities demonstrating proper use of Optional returns.

This module shows best practices for functions that return Optional
values, with explicit return None statements.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import re

from genomevault.types import ValidationResult, GenomicAnnotation
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


def validate_chromosome_name(chrom: str) -> Optional[str]:
    """
    Validate and normalize chromosome name.

    Args:
        chrom: Chromosome name to validate

    Returns:
        Normalized chromosome name or None if invalid
    """
    if not chrom:
        return None

    # Normalize common formats
    chrom = chrom.upper().strip()

    # Handle numeric chromosomes
    if chrom.isdigit():
        num = int(chrom)
        if 1 <= num <= 22:
            return f"chr{num}"
        return None  # Explicit return for invalid numeric

    # Handle chr prefix
    if chrom.startswith("CHR"):
        suffix = chrom[3:]
        if suffix.isdigit():
            num = int(suffix)
            if 1 <= num <= 22:
                return f"chr{num}"
            return None  # Explicit return for out of range
        elif suffix in ["X", "Y", "M", "MT"]:
            return f"chr{suffix}"
        return None  # Explicit return for invalid suffix

    # Handle special chromosomes
    if chrom in ["X", "Y", "M", "MT"]:
        return f"chr{chrom}"

    return None  # Explicit return for unrecognized format


def find_gene_annotation(
    position: int, annotations: List[GenomicAnnotation]
) -> Optional[GenomicAnnotation]:
    """
    Find annotation for a genomic position.

    Args:
        position: Genomic position
        annotations: List of annotations to search

    Returns:
        Matching annotation or None if not found
    """
    if not annotations:
        return None  # Explicit return for empty list

    for annotation in annotations:
        # Assuming annotation has position info (simplified)
        if annotation.get("gene_name"):
            # Simplified check - in reality would check position ranges
            return annotation

    return None  # Explicit return when no match found


def parse_variant_string(variant: str) -> Optional[Dict[str, Any]]:
    """
    Parse variant string in format 'chr:pos ref>alt'.

    Args:
        variant: Variant string

    Returns:
        Parsed variant dict or None if invalid
    """
    if not variant:
        return None  # Explicit return for empty string

    # Pattern: chr1:123456 A>G
    pattern = r"^(chr[\dXYMT]+):(\d+)\s+([ACGT]+)>([ACGT]+)$"
    match = re.match(pattern, variant, re.IGNORECASE)

    if not match:
        logger.warning(f"Invalid variant format: {variant}")
        return None  # Explicit return for no match

    try:
        chrom, pos, ref, alt = match.groups()
        return {
            "chromosome": chrom.upper(),
            "position": int(pos),
            "ref": ref.upper(),
            "alt": alt.upper(),
        }
    except (ValueError, AttributeError) as e:
        logger.error(f"Error parsing variant: {e}")
        return None  # Explicit return on parse error


def load_config_file(path: Path) -> Optional[Dict[str, Any]]:
    """
    Load configuration from file.

    Args:
        path: Path to config file

    Returns:
        Configuration dict or None if file doesn't exist or is invalid
    """
    if not path.exists():
        logger.debug(f"Config file not found: {path}")
        return None  # Explicit return for missing file

    if not path.is_file():
        logger.warning(f"Path is not a file: {path}")
        return None  # Explicit return for non-file

    try:
        import json

        with open(path, "r") as f:
            config = json.load(f)

        if not isinstance(config, dict):
            logger.error(f"Config is not a dictionary: {type(config)}")
            return None  # Explicit return for wrong type

        return config

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return None  # Explicit return for invalid JSON
    except IOError as e:
        logger.error(f"Error reading config file: {e}")
        return None  # Explicit return for IO error


def get_quality_score(scores: List[float], min_threshold: float = 0.0) -> Optional[float]:
    """
    Calculate average quality score.

    Args:
        scores: List of quality scores
        min_threshold: Minimum acceptable average

    Returns:
        Average score or None if below threshold or invalid
    """
    if not scores:
        return None  # Explicit return for empty list

    try:
        avg = sum(scores) / len(scores)

        if avg < min_threshold:
            logger.debug(f"Score {avg} below threshold {min_threshold}")
            return None  # Explicit return for below threshold

        return avg

    except (TypeError, ZeroDivisionError) as e:
        logger.error(f"Error calculating average: {e}")
        return None  # Explicit return on calculation error


def find_file_in_directory(
    directory: Path, pattern: str, recursive: bool = False
) -> Optional[Path]:
    """
    Find first file matching pattern in directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        recursive: Whether to search recursively

    Returns:
        Path to first matching file or None if not found
    """
    if not directory.exists():
        logger.debug(f"Directory does not exist: {directory}")
        return None  # Explicit return for missing directory

    if not directory.is_dir():
        logger.warning(f"Path is not a directory: {directory}")
        return None  # Explicit return for non-directory

    try:
        if recursive:
            matches = list(directory.rglob(pattern))
        else:
            matches = list(directory.glob(pattern))

        if not matches:
            logger.debug(f"No files matching {pattern} in {directory}")
            return None  # Explicit return for no matches

        return matches[0]

    except (OSError, ValueError) as e:
        logger.error(f"Error searching directory: {e}")
        return None  # Explicit return on search error


def validate_data_batch(
    data: List[Dict[str, Any]], required_fields: List[str]
) -> Optional[ValidationResult]:
    """
    Validate a batch of data records.

    Args:
        data: List of data records
        required_fields: Fields that must be present

    Returns:
        Validation result or None if input is invalid
    """
    if not data:
        logger.warning("Empty data batch provided")
        return None  # Explicit return for empty data

    if not required_fields:
        logger.warning("No required fields specified")
        return None  # Explicit return for no requirements

    errors = []
    warnings = []
    failed = 0

    for i, record in enumerate(data):
        if not isinstance(record, dict):
            errors.append(f"Record {i} is not a dictionary")
            failed += 1
            continue

        for field in required_fields:
            if field not in record:
                errors.append(f"Record {i} missing field: {field}")
                failed += 1
                break

    result: ValidationResult = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "processed_items": len(data),
        "failed_items": failed,
    }

    return result


# Example of a function that should NOT return None
def calculate_hash(data: bytes) -> str:
    """
    Calculate SHA256 hash of data.

    This function always returns a value, never None.

    Args:
        data: Data to hash

    Returns:
        Hex digest of hash
    """
    import hashlib

    if not data:
        # Return hash of empty bytes, not None
        return hashlib.sha256(b"").hexdigest()

    return hashlib.sha256(data).hexdigest()
    # No explicit return None needed - function always returns string
