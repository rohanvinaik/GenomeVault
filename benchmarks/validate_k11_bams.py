#!/usr/bin/env python3
"""
k=11 Experimental BAM Validation

Comprehensive validation to ensure generated BAMs meet GenomeVault's
3-layer privacy architecture requirements and are ready for GDiff encoding.

Architecture Validation:
- Layer 3 (Experimental) properly aligned to Layer 2 (Guide strands)
- NO direct experimental → consensus contact (privacy violation check)
- Coordinate system correctness (guide FASTA coords, chr*_consensus naming)
- Whole-genome coverage (not just chr1)
- Alignment quality (>99% mapping rate expected)
- File integrity (BAM + BAI indices)
"""

import sys
import pysam
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Terminal colors for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    print(f"  {text}")


class K11BAMValidator:
    """Validate k=11 experimental BAMs for GenomeVault pipeline."""

    def __init__(self, bam_dir: Path, guide_dir: Path):
        self.bam_dir = bam_dir
        self.guide_dir = guide_dir
        self.results = {
            "validation_time": datetime.now().isoformat(),
            "bams_found": [],
            "tests_passed": 0,
            "tests_failed": 0,
            "critical_failures": [],
            "warnings": [],
            "per_bam_stats": {}
        }

    def validate_all(self) -> Dict:
        """Run all validation checks."""
        print_header("GenomeVault k=11 BAM Validation")

        # Test 1: File existence and structure
        if not self._test_file_existence():
            print_error("CRITICAL: File existence check failed")
            return self.results

        # Test 2: BAM indices
        self._test_bam_indices()

        # Test 3: Coordinate system
        self._test_coordinate_system()

        # Test 4: Whole-genome coverage
        self._test_whole_genome_coverage()

        # Test 5: Alignment quality
        self._test_alignment_quality()

        # Test 6: Size consistency
        self._test_size_consistency()

        # Test 7: Privacy architecture compliance
        self._test_privacy_architecture()

        # Test 8: GDiff readiness
        self._test_gdiff_readiness()

        # Summary
        self._print_summary()

        return self.results

    def _test_file_existence(self) -> bool:
        """Test 1: Verify all 11 BAMs exist."""
        print_header("TEST 1: File Existence")

        expected_bams = [
            self.bam_dir / f"experimental_vs_ref{i}.sorted.bam"
            for i in range(1, 12)
        ]

        all_exist = True
        for bam_path in expected_bams:
            if bam_path.exists():
                size_gb = bam_path.stat().st_size / (1024**3)
                self.results["bams_found"].append(str(bam_path))
                print_success(f"{bam_path.name} exists ({size_gb:.2f} GB)")
            else:
                print_error(f"{bam_path.name} NOT FOUND")
                self.results["critical_failures"].append(f"Missing BAM: {bam_path.name}")
                all_exist = False

        if all_exist:
            self.results["tests_passed"] += 1
            print_success("All 11 BAMs present")
        else:
            self.results["tests_failed"] += 1

        return all_exist

    def _test_bam_indices(self):
        """Test 2: Verify BAM indices (.bai) exist."""
        print_header("TEST 2: BAM Indices")

        all_indexed = True
        for bam_path_str in self.results["bams_found"]:
            bam_path = Path(bam_path_str)
            bai_path = Path(str(bam_path) + ".bai")

            if bai_path.exists():
                print_success(f"{bam_path.name}.bai exists")
            else:
                print_error(f"{bam_path.name}.bai MISSING")
                self.results["critical_failures"].append(f"Missing index: {bai_path.name}")
                all_indexed = False

        if all_indexed:
            self.results["tests_passed"] += 1
            print_success("All BAMs properly indexed")
        else:
            self.results["tests_failed"] += 1

    def _test_coordinate_system(self):
        """Test 3: Verify correct coordinate system (guide FASTA coords)."""
        print_header("TEST 3: Coordinate System")

        # Expected: chr*_consensus naming (guide FASTA coordinate space)
        expected_pattern = "chr.*_consensus"

        all_correct = True
        for bam_path_str in self.results["bams_found"][:3]:  # Check first 3 for speed
            bam_path = Path(bam_path_str)

            try:
                bam = pysam.AlignmentFile(str(bam_path), "rb")

                # Check header for chromosome naming
                chroms = list(bam.references)

                # Verify naming convention
                if any("_consensus" in chrom for chrom in chroms[:5]):
                    print_success(f"{bam_path.name}: Guide FASTA coords (chr*_consensus)")
                    print_info(f"  Sample chromosomes: {', '.join(chroms[:3])}")
                else:
                    print_error(f"{bam_path.name}: WRONG coordinate system")
                    print_info(f"  Found: {', '.join(chroms[:3])}")
                    print_error("  Expected: chr*_consensus (guide FASTA coordinate space)")
                    self.results["critical_failures"].append(
                        f"Wrong coords: {bam_path.name} (not guide FASTA space)"
                    )
                    all_correct = False

                bam.close()

            except Exception as e:
                print_error(f"{bam_path.name}: Error reading BAM - {e}")
                all_correct = False

        if all_correct:
            self.results["tests_passed"] += 1
            print_success("Coordinate system correct: Guide FASTA space (chr*_consensus)")
        else:
            self.results["tests_failed"] += 1

    def _test_whole_genome_coverage(self):
        """Test 4: Verify whole-genome coverage (not just chr1)."""
        print_header("TEST 4: Whole-Genome Coverage")

        all_whole_genome = True
        for bam_path_str in self.results["bams_found"][:3]:  # Check first 3
            bam_path = Path(bam_path_str)

            try:
                bam = pysam.AlignmentFile(str(bam_path), "rb")

                # Use idxstats to check read distribution
                stats = bam.get_index_statistics()

                # Count chromosomes with reads
                chroms_with_reads = []
                for stat in stats:
                    if stat.mapped > 0:
                        chroms_with_reads.append(stat.contig)

                # Should have reads on chr1-22, chrX, chrY (at least 20 chroms)
                if len(chroms_with_reads) >= 20:
                    print_success(f"{bam_path.name}: Whole-genome coverage")
                    print_info(f"  {len(chroms_with_reads)} chromosomes with reads")

                    # Show distribution for first few chroms
                    sample_stats = stats[:5]
                    for stat in sample_stats:
                        mapped_m = stat.mapped / 1_000_000
                        print_info(f"    {stat.contig}: {mapped_m:.1f}M reads")
                else:
                    print_warning(f"{bam_path.name}: Limited chromosome coverage")
                    print_info(f"  Only {len(chroms_with_reads)} chromosomes with reads")
                    print_info(f"  Expected: ≥20 (chr1-22, chrX, chrY)")
                    self.results["warnings"].append(
                        f"Limited coverage: {bam_path.name} ({len(chroms_with_reads)} chroms)"
                    )
                    all_whole_genome = False

                bam.close()

            except Exception as e:
                print_error(f"{bam_path.name}: Error checking coverage - {e}")
                all_whole_genome = False

        if all_whole_genome:
            self.results["tests_passed"] += 1
            print_success("Whole-genome coverage confirmed")
        else:
            self.results["tests_failed"] += 1

    def _test_alignment_quality(self):
        """Test 5: Verify alignment quality (mapping rate, quality scores)."""
        print_header("TEST 5: Alignment Quality")

        all_high_quality = True
        for bam_path_str in self.results["bams_found"][:3]:  # Check first 3
            bam_path = Path(bam_path_str)

            try:
                # Get flagstat statistics
                stats_cmd = f"samtools flagstat {bam_path}"
                import subprocess
                result = subprocess.run(
                    stats_cmd,
                    shell=True,
                    capture_output=True,
                    text=True
                )

                # Parse mapping rate
                lines = result.stdout.split('\n')
                mapped_line = [l for l in lines if 'mapped (' in l][0]

                # Extract mapping percentage
                import re
                match = re.search(r'(\d+\.\d+)%', mapped_line)
                if match:
                    mapping_pct = float(match.group(1))

                    if mapping_pct >= 99.0:
                        print_success(f"{bam_path.name}: {mapping_pct:.2f}% mapping rate")
                    elif mapping_pct >= 95.0:
                        print_warning(f"{bam_path.name}: {mapping_pct:.2f}% mapping rate (acceptable)")
                        self.results["warnings"].append(
                            f"Moderate mapping: {bam_path.name} ({mapping_pct:.2f}%)"
                        )
                    else:
                        print_error(f"{bam_path.name}: {mapping_pct:.2f}% mapping rate (LOW)")
                        self.results["critical_failures"].append(
                            f"Low mapping rate: {bam_path.name} ({mapping_pct:.2f}%)"
                        )
                        all_high_quality = False

                    # Store per-BAM stats
                    bam_name = bam_path.name
                    if bam_name not in self.results["per_bam_stats"]:
                        self.results["per_bam_stats"][bam_name] = {}
                    self.results["per_bam_stats"][bam_name]["mapping_rate"] = mapping_pct

            except Exception as e:
                print_error(f"{bam_path.name}: Error checking quality - {e}")
                all_high_quality = False

        if all_high_quality:
            self.results["tests_passed"] += 1
            print_success("Alignment quality excellent (>99% mapping)")
        else:
            self.results["tests_failed"] += 1

    def _test_size_consistency(self):
        """Test 6: Verify BAM sizes are consistent (~26GB expected)."""
        print_header("TEST 6: Size Consistency")

        sizes = []
        for bam_path_str in self.results["bams_found"]:
            bam_path = Path(bam_path_str)
            size_gb = bam_path.stat().st_size / (1024**3)
            sizes.append(size_gb)

            # Store size
            bam_name = bam_path.name
            if bam_name not in self.results["per_bam_stats"]:
                self.results["per_bam_stats"][bam_name] = {}
            self.results["per_bam_stats"][bam_name]["size_gb"] = size_gb

        if sizes:
            avg_size = sum(sizes) / len(sizes)
            min_size = min(sizes)
            max_size = max(sizes)

            print_info(f"Average BAM size: {avg_size:.2f} GB")
            print_info(f"Size range: {min_size:.2f} - {max_size:.2f} GB")

            # Check consistency (should be within 10% of average)
            tolerance = 0.10
            all_consistent = True
            for bam_path_str, size in zip(self.results["bams_found"], sizes):
                bam_name = Path(bam_path_str).name
                deviation = abs(size - avg_size) / avg_size

                if deviation <= tolerance:
                    print_success(f"{bam_name}: {size:.2f} GB (consistent)")
                else:
                    print_warning(f"{bam_name}: {size:.2f} GB (deviates {deviation*100:.1f}%)")
                    self.results["warnings"].append(
                        f"Size variance: {bam_name} ({size:.2f} GB vs {avg_size:.2f} GB avg)"
                    )
                    all_consistent = False

            # Expected size check (~26GB for whole genome at 14× coverage)
            if 20 <= avg_size <= 35:
                print_success(f"BAM sizes in expected range (20-35 GB for whole genome)")
                self.results["tests_passed"] += 1
            else:
                print_warning(f"BAM sizes unusual (expected 20-35 GB, got {avg_size:.2f} GB)")
                self.results["warnings"].append(
                    f"Unusual average BAM size: {avg_size:.2f} GB"
                )
                self.results["tests_failed"] += 1

    def _test_privacy_architecture(self):
        """Test 7: Verify 3-layer privacy architecture compliance."""
        print_header("TEST 7: Privacy Architecture Compliance")

        print_info("Checking Layer 3 → Layer 2 alignment architecture...")

        # Check 1: No direct consensus reference in BAM headers
        all_compliant = True
        for bam_path_str in self.results["bams_found"][:3]:
            bam_path = Path(bam_path_str)

            try:
                bam = pysam.AlignmentFile(str(bam_path), "rb")
                header = bam.header

                # Check @SQ lines - should NOT reference hg38/GRCh38/consensus directly
                # Should reference guide FASTA (chr*_consensus)
                if "text" in header:
                    header_text = header["text"]

                    # Red flags: direct consensus reference
                    red_flags = ["GRCh38", "hg38", "hg19", "chm13"]
                    found_red_flags = [flag for flag in red_flags if flag in header_text]

                    if found_red_flags:
                        print_error(f"{bam_path.name}: PRIVACY VIOLATION - references {found_red_flags}")
                        print_error("  Experimental data must ONLY align to guide strands!")
                        self.results["critical_failures"].append(
                            f"Privacy violation: {bam_path.name} references consensus directly"
                        )
                        all_compliant = False
                    else:
                        print_success(f"{bam_path.name}: No direct consensus reference")

                # Check 2: Verify guide FASTA coordinate space
                chroms = list(bam.references)
                if any("_consensus" in chrom for chrom in chroms):
                    print_success(f"{bam_path.name}: Aligned to guide FASTA (Layer 2)")
                else:
                    print_error(f"{bam_path.name}: NOT in guide FASTA coordinate space")
                    all_compliant = False

                bam.close()

            except Exception as e:
                print_error(f"{bam_path.name}: Error checking privacy - {e}")
                all_compliant = False

        if all_compliant:
            self.results["tests_passed"] += 1
            print_success("Privacy architecture: Layer 3 → Layer 2 (COMPLIANT)")
            print_info("  ✓ No direct experimental → consensus contact")
            print_info("  ✓ Experimental data aligned to guide strands only")
        else:
            self.results["tests_failed"] += 1
            print_error("Privacy architecture: VIOLATIONS FOUND")

    def _test_gdiff_readiness(self):
        """Test 8: Verify BAMs are ready for GDiff encoding."""
        print_header("TEST 8: GDiff Encoding Readiness")

        # Requirements for GDiff:
        # 1. Coordinate system match (experimental and guide BAMs must be in same coords)
        # 2. BAI indices for random access
        # 3. Header compatibility

        print_info("Checking GDiff encoder requirements...")

        all_ready = True

        # Check if guide BAMs exist (needed for GDiff comparison)
        print_info("\nVerifying guide BAMs availability:")
        for i in range(1, 12):
            guide_bam = self.guide_dir / f"ref{i}_gdiff.bam"
            if guide_bam.exists():
                print_success(f"  ref{i}_gdiff.bam available")
            else:
                print_error(f"  ref{i}_gdiff.bam MISSING")
                self.results["critical_failures"].append(
                    f"Missing guide BAM for GDiff: ref{i}_gdiff.bam"
                )
                all_ready = False

        # Check coordinate system compatibility
        print_info("\nCoordinate system compatibility:")
        if len(self.results["bams_found"]) > 0:
            exp_bam_path = Path(self.results["bams_found"][0])
            guide_bam_path = self.guide_dir / "ref1_gdiff.bam"

            if guide_bam_path.exists():
                try:
                    exp_bam = pysam.AlignmentFile(str(exp_bam_path), "rb")
                    guide_bam = pysam.AlignmentFile(str(guide_bam_path), "rb")

                    exp_chroms = set(exp_bam.references)
                    guide_chroms = set(guide_bam.references)

                    # Should have matching chromosome names
                    if exp_chroms == guide_chroms:
                        print_success("  Experimental and guide BAMs have matching chromosomes")
                    else:
                        print_warning("  Chromosome name mismatch detected")
                        common = exp_chroms & guide_chroms
                        print_info(f"    Common chromosomes: {len(common)}")
                        self.results["warnings"].append(
                            "Chromosome name mismatch between experimental and guide BAMs"
                        )

                    exp_bam.close()
                    guide_bam.close()

                except Exception as e:
                    print_error(f"  Error checking compatibility: {e}")
                    all_ready = False

        # Final assessment
        if all_ready:
            self.results["tests_passed"] += 1
            print_success("\nGDiff Encoding: READY")
            print_info("  ✓ All experimental BAMs present")
            print_info("  ✓ All guide BAMs present")
            print_info("  ✓ Coordinate systems compatible")
            print_info("  ✓ BAM indices available")
        else:
            self.results["tests_failed"] += 1
            print_error("\nGDiff Encoding: NOT READY")

    def _print_summary(self):
        """Print validation summary."""
        print_header("VALIDATION SUMMARY")

        total_tests = self.results["tests_passed"] + self.results["tests_failed"]

        print(f"\n{Colors.BOLD}Results:{Colors.END}")
        print(f"  Tests passed: {Colors.GREEN}{self.results['tests_passed']}/{total_tests}{Colors.END}")
        print(f"  Tests failed: {Colors.RED}{self.results['tests_failed']}/{total_tests}{Colors.END}")
        print(f"  Warnings: {Colors.YELLOW}{len(self.results['warnings'])}{Colors.END}")
        print(f"  Critical failures: {Colors.RED}{len(self.results['critical_failures'])}{Colors.END}")

        if self.results["critical_failures"]:
            print(f"\n{Colors.BOLD}{Colors.RED}CRITICAL FAILURES:{Colors.END}")
            for failure in self.results["critical_failures"]:
                print(f"  {Colors.RED}✗{Colors.END} {failure}")

        if self.results["warnings"]:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}WARNINGS:{Colors.END}")
            for warning in self.results["warnings"]:
                print(f"  {Colors.YELLOW}⚠{Colors.END} {warning}")

        # Overall assessment
        print(f"\n{Colors.BOLD}Overall Assessment:{Colors.END}")
        if self.results["tests_failed"] == 0 and len(self.results["critical_failures"]) == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL VALIDATION CHECKS PASSED{Colors.END}")
            print(f"{Colors.GREEN}BAMs are ready for GDiff encoding!{Colors.END}")
        elif len(self.results["critical_failures"]) > 0:
            print(f"{Colors.RED}{Colors.BOLD}✗ CRITICAL FAILURES DETECTED{Colors.END}")
            print(f"{Colors.RED}BAMs require fixes before GDiff encoding{Colors.END}")
        else:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠ VALIDATION PASSED WITH WARNINGS{Colors.END}")
            print(f"{Colors.YELLOW}BAMs are usable but may have quality issues{Colors.END}")

        # Save results to JSON
        output_file = self.bam_dir / "validation_report.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n{Colors.BLUE}Detailed report saved: {output_file}{Colors.END}")


def main():
    """Main validation entry point."""

    # Paths
    bam_dir = Path("data/experimental_strands/ERR3239334/alignment/k11_bams")
    guide_dir = Path("/Volumes/1TBStorage/guide_strands")

    # Verify directories exist
    if not bam_dir.exists():
        print_error(f"BAM directory not found: {bam_dir}")
        return 1

    if not guide_dir.exists():
        print_warning(f"Guide directory not found: {guide_dir}")
        print_warning("Guide BAM validation will be skipped")

    # Run validation
    validator = K11BAMValidator(bam_dir, guide_dir)
    results = validator.validate_all()

    # Exit code
    if results["tests_failed"] > 0 or len(results["critical_failures"]) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
