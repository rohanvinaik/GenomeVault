#!/usr/bin/env python3
"""
Validate Reference Pool FASTQ Files

Checks integrity and completeness of generated reference pool:
- 3 reference genomes (ref1, ref2, ref3)
- 1 query/experimental genome
- FASTQ file format validation
- Read count verification
- Quality metrics
"""

import sys
import gzip
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_fastq_format(fastq_path: Path, sample_size: int = 1000) -> dict:
    """
    Validate FASTQ file format and extract basic metrics.

    Args:
        fastq_path: Path to FASTQ file (can be gzipped)
        sample_size: Number of reads to sample for quality check

    Returns:
        dict with validation results and metrics
    """
    try:
        opener = gzip.open if str(fastq_path).endswith('.gz') else open

        total_reads = 0
        line_count = 0
        quality_scores = []
        read_lengths = []

        with opener(fastq_path, 'rt') as f:
            current_read = []
            for line in f:
                line_count += 1
                current_read.append(line.strip())

                if len(current_read) == 4:
                    # Validate FASTQ format
                    header, sequence, plus, quality = current_read

                    # Check header
                    if not header.startswith('@'):
                        return {
                            'valid': False,
                            'error': f'Invalid header at line {line_count - 3}: {header[:50]}'
                        }

                    # Check separator
                    if not plus.startswith('+'):
                        return {
                            'valid': False,
                            'error': f'Invalid separator at line {line_count - 1}: {plus[:50]}'
                        }

                    # Check sequence and quality length match
                    if len(sequence) != len(quality):
                        return {
                            'valid': False,
                            'error': f'Sequence/quality length mismatch: {len(sequence)} != {len(quality)}'
                        }

                    # Collect metrics
                    total_reads += 1
                    if total_reads <= sample_size:
                        read_lengths.append(len(sequence))
                        # Average quality (Phred+33 encoding)
                        avg_q = sum(ord(c) - 33 for c in quality) / len(quality)
                        quality_scores.append(avg_q)

                    current_read = []

                    # Early exit for quick validation
                    if total_reads >= sample_size and sample_size > 0:
                        # Count remaining reads
                        for remaining_line in f:
                            line_count += 1
                        total_reads = line_count // 4
                        break

        # Calculate metrics
        avg_read_length = sum(read_lengths) / len(read_lengths) if read_lengths else 0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        return {
            'valid': True,
            'total_reads': total_reads,
            'avg_read_length': avg_read_length,
            'avg_quality': avg_quality,
            'file_size_mb': fastq_path.stat().st_size / (1024 * 1024)
        }

    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


def validate_paired_end(r1_path: Path, r2_path: Path) -> dict:
    """Validate paired-end FASTQ files match."""
    r1_result = validate_fastq_format(r1_path)
    r2_result = validate_fastq_format(r2_path)

    if not r1_result['valid']:
        return {'valid': False, 'error': f"R1 invalid: {r1_result['error']}"}

    if not r2_result['valid']:
        return {'valid': False, 'error': f"R2 invalid: {r2_result['error']}"}

    # Check read counts match
    if r1_result['total_reads'] != r2_result['total_reads']:
        return {
            'valid': False,
            'error': f"Read count mismatch: R1={r1_result['total_reads']}, R2={r2_result['total_reads']}"
        }

    return {
        'valid': True,
        'total_reads': r1_result['total_reads'],
        'r1_metrics': r1_result,
        'r2_metrics': r2_result
    }


def validate_reference_pool(pool_dir: Path) -> dict:
    """
    Validate complete reference pool structure and integrity.

    Expected structure:
        pool_dir/
        ├── references/
        │   ├── ref1/
        │   │   ├── sample1_r1.fastq.gz
        │   │   ├── sample1_r2.fastq.gz
        │   │   ├── variants_snp.vcf
        │   │   └── variants_indel.vcf
        │   ├── ref2/
        │   │   ├── sample2_r1.fastq.gz
        │   │   ├── sample2_r2.fastq.gz
        │   │   ├── variants_snp.vcf
        │   │   └── variants_indel.vcf
        │   └── ref3/
        │       ├── sample3_r1.fastq.gz
        │       ├── sample3_r2.fastq.gz
        │       ├── variants_snp.vcf
        │       └── variants_indel.vcf
        └── query/
            ├── sample4_r1.fastq.gz
            ├── sample4_r2.fastq.gz
            ├── variants_snp.vcf
            └── variants_indel.vcf
    """
    results = {
        'valid': True,
        'references': {},
        'query': None,
        'k_anonymity': 0,
        'errors': []
    }

    references_dir = pool_dir / "references"
    query_dir = pool_dir / "query"

    # Check directories exist
    if not references_dir.exists():
        results['valid'] = False
        results['errors'].append(f"References directory not found: {references_dir}")
        return results

    # Validate each reference
    for ref_num in [1, 2, 3]:
        ref_dir = references_dir / f"ref{ref_num}"

        if not ref_dir.exists():
            logger.warning(f"Reference {ref_num} directory not found: {ref_dir}")
            results['references'][f'ref{ref_num}'] = {
                'valid': False,
                'error': 'Directory not found'
            }
            continue

        # Check FASTQ files
        r1 = ref_dir / f"sample{ref_num}_r1.fastq.gz"
        r2 = ref_dir / f"sample{ref_num}_r2.fastq.gz"
        snp_vcf = ref_dir / "variants_snp.vcf"
        indel_vcf = ref_dir / "variants_indel.vcf"

        missing = []
        if not r1.exists():
            missing.append(r1.name)
        if not r2.exists():
            missing.append(r2.name)
        if not snp_vcf.exists():
            missing.append(snp_vcf.name)
        if not indel_vcf.exists():
            missing.append(indel_vcf.name)

        if missing:
            results['references'][f'ref{ref_num}'] = {
                'valid': False,
                'error': f'Missing files: {", ".join(missing)}'
            }
            continue

        # Validate FASTQ format and pairing
        paired_result = validate_paired_end(r1, r2)

        if paired_result['valid']:
            results['references'][f'ref{ref_num}'] = {
                'valid': True,
                'total_reads': paired_result['total_reads'],
                'r1_size_mb': paired_result['r1_metrics']['file_size_mb'],
                'r2_size_mb': paired_result['r2_metrics']['file_size_mb'],
                'avg_read_length': paired_result['r1_metrics']['avg_read_length'],
                'avg_quality': paired_result['r1_metrics']['avg_quality'],
                'has_variants': snp_vcf.exists() and indel_vcf.exists()
            }
            results['k_anonymity'] += 1
        else:
            results['valid'] = False
            results['references'][f'ref{ref_num}'] = paired_result
            results['errors'].append(f"Reference {ref_num}: {paired_result['error']}")

    # Validate query sample
    if query_dir.exists():
        r1 = query_dir / "sample4_r1.fastq.gz"
        r2 = query_dir / "sample4_r2.fastq.gz"
        snp_vcf = query_dir / "variants_snp.vcf"
        indel_vcf = query_dir / "variants_indel.vcf"

        if r1.exists() and r2.exists():
            paired_result = validate_paired_end(r1, r2)

            if paired_result['valid']:
                results['query'] = {
                    'valid': True,
                    'total_reads': paired_result['total_reads'],
                    'r1_size_mb': paired_result['r1_metrics']['file_size_mb'],
                    'r2_size_mb': paired_result['r2_metrics']['file_size_mb'],
                    'avg_read_length': paired_result['r1_metrics']['avg_read_length'],
                    'avg_quality': paired_result['r1_metrics']['avg_quality'],
                    'has_variants': snp_vcf.exists() and indel_vcf.exists()
                }
            else:
                results['valid'] = False
                results['query'] = paired_result
                results['errors'].append(f"Query sample: {paired_result['error']}")
        else:
            results['query'] = {
                'valid': False,
                'error': 'FASTQ files not found'
            }
    else:
        results['query'] = {
            'valid': False,
            'error': 'Query directory not found'
        }

    return results


def main():
    """Validate reference pool and generate report."""
    logger.info("=" * 70)
    logger.info("Reference Pool Validation")
    logger.info("=" * 70)

    pool_dir = Path("/Users/rohanvinaik/genomevault/benchmark_results/differential_encoding_samples")

    if not pool_dir.exists():
        logger.error(f"Pool directory not found: {pool_dir}")
        return 1

    logger.info(f"Validating pool directory: {pool_dir}")
    logger.info("")

    results = validate_reference_pool(pool_dir)

    # Print results
    logger.info("=" * 70)
    logger.info("Validation Results")
    logger.info("=" * 70)

    # References
    logger.info("\n📚 REFERENCE GENOMES:")
    ref_count = 0
    for ref_id in ['ref1', 'ref2', 'ref3']:
        if ref_id in results['references']:
            ref = results['references'][ref_id]
            if ref['valid']:
                logger.info(f"\n✅ {ref_id.upper()}:")
                logger.info(f"   Reads: {ref['total_reads']:,}")
                logger.info(f"   Read Length: {ref['avg_read_length']:.1f} bp")
                logger.info(f"   Quality Score: {ref['avg_quality']:.1f}")
                logger.info(f"   R1 Size: {ref['r1_size_mb']:.1f} MB")
                logger.info(f"   R2 Size: {ref['r2_size_mb']:.1f} MB")
                logger.info(f"   Has Variants: {'Yes' if ref['has_variants'] else 'No'}")
                ref_count += 1
            else:
                logger.warning(f"\n❌ {ref_id.upper()}: {ref.get('error', 'Unknown error')}")

    # Query
    logger.info("\n🔬 QUERY/EXPERIMENTAL GENOME:")
    if results['query'] and results['query']['valid']:
        q = results['query']
        logger.info(f"\n✅ QUERY:")
        logger.info(f"   Reads: {q['total_reads']:,}")
        logger.info(f"   Read Length: {q['avg_read_length']:.1f} bp")
        logger.info(f"   Quality Score: {q['avg_quality']:.1f}")
        logger.info(f"   R1 Size: {q['r1_size_mb']:.1f} MB")
        logger.info(f"   R2 Size: {q['r2_size_mb']:.1f} MB")
        logger.info(f"   Has Variants: {'Yes' if q['has_variants'] else 'No'}")
    else:
        logger.warning(f"\n❌ QUERY: {results['query'].get('error', 'Not found') if results['query'] else 'Not found'}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Valid References: {ref_count}/3")
    logger.info(f"Valid Query: {'Yes' if results['query'] and results['query']['valid'] else 'No'}")
    logger.info(f"k-Anonymity: k={results['k_anonymity']}")

    if results['errors']:
        logger.info("\n⚠️  ERRORS:")
        for error in results['errors']:
            logger.info(f"  - {error}")

    # Overall status
    logger.info("")
    if ref_count == 3 and results['query'] and results['query']['valid']:
        logger.info("🎉 ✅ VALIDATION PASSED - Complete reference pool with k=3")
        logger.info("Ready for FASTQ → differential encoding pipeline")
        return 0
    elif ref_count >= 2:
        logger.info(f"⚠️  PARTIAL - {ref_count} references available (minimum k={ref_count})")
        if not (results['query'] and results['query']['valid']):
            logger.info("❌ Query sample missing or invalid")
        return 2
    else:
        logger.info("❌ VALIDATION FAILED - Insufficient samples")
        logger.info(f"   Only {ref_count}/3 references valid")
        return 1


if __name__ == "__main__":
    sys.exit(main())
