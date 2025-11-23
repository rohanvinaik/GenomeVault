"""
genomevault/clinical_db/data_acquisition.py

Automated pipeline to download and process clinical variant databases
"""

import gzip
import requests
import logging
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin
import time

from .database import (
    ClinicalSNP, ClinicalCondition, ClinicalAnnotation,
    PopulationFrequency, FunctionalImpact, ClinicalDatabaseBuilder
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinVarDownloader:
    """
    Download and parse ClinVar variants
    
    ClinVar: NCBI database of relationships between variants and phenotypes
    URL: https://www.ncbi.nlm.nih.gov/clinvar/
    """
    
    CLINVAR_FTP = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/"
    CLINVAR_VCF_GRCH38 = "vcf_GRCh38/clinvar.vcf.gz"
    CLINVAR_VCF_GRCH37 = "vcf_GRCh37/clinvar.vcf.gz"
    
    # Clinical significance mapping
    PATHOGENIC_TERMS = {
        'Pathogenic', 'Likely_pathogenic', 'Pathogenic/Likely_pathogenic'
    }
    
    BENIGN_TERMS = {
        'Benign', 'Likely_benign', 'Benign/Likely_benign'
    }
    
    def __init__(self, genome_build: str = "GRCh38", output_dir: str = "data/raw"):
        self.genome_build = genome_build
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_vcf(self) -> Path:
        """Download ClinVar VCF file"""
        vcf_path = self.CLINVAR_VCF_GRCH38 if self.genome_build == "GRCh38" else self.CLINVAR_VCF_GRCH37
        url = urljoin(self.CLINVAR_FTP, vcf_path)
        
        output_file = self.output_dir / f"clinvar_{self.genome_build}.vcf.gz"
        
        if output_file.exists():
            logger.info(f"ClinVar VCF already exists: {output_file}")
            return output_file
        
        logger.info(f"Downloading ClinVar VCF from {url}")
        logger.info(f"This may take 10-15 minutes (~2GB file)...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size:
                    progress = (downloaded / total_size) * 100
                    if downloaded % (10 * 1024 * 1024) == 0:  # Log every 10MB
                        logger.info(f"Downloaded {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({progress:.1f}%)")
        
        logger.info(f"Download complete: {output_file}")
        return output_file
    
    def parse_vcf(self, vcf_path: Path, 
                  filter_pathogenic: bool = True,
                  min_review_stars: int = 1) -> List[ClinicalSNP]:
        """
        Parse ClinVar VCF file and extract clinical SNPs
        
        Args:
            vcf_path: Path to ClinVar VCF file
            filter_pathogenic: Only include pathogenic/likely pathogenic variants
            min_review_stars: Minimum review status stars (0-4)
            
        Returns:
            List of ClinicalSNP objects
        """
        logger.info(f"Parsing ClinVar VCF: {vcf_path}")
        
        snps = []
        total_variants = 0
        filtered_count = 0
        
        with gzip.open(vcf_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                total_variants += 1
                if total_variants % 10000 == 0:
                    logger.info(f"Processed {total_variants} variants, kept {len(snps)}")
                
                try:
                    snp = self._parse_vcf_line(line)
                    
                    if snp is None:
                        continue
                    
                    # Apply filters
                    if filter_pathogenic and not snp.is_pathogenic():
                        filtered_count += 1
                        continue
                    
                    if snp.clinical_annotations and snp.clinical_annotations.stars < min_review_stars:
                        filtered_count += 1
                        continue
                    
                    snps.append(snp)
                    
                except Exception as e:
                    logger.warning(f"Error parsing line: {e}")
                    continue
        
        logger.info(f"Parsing complete: {len(snps)} SNPs kept from {total_variants} total variants")
        logger.info(f"Filtered out: {filtered_count} variants")
        
        return snps
    
    def _parse_vcf_line(self, line: str) -> Optional[ClinicalSNP]:
        """Parse single VCF line into ClinicalSNP"""
        fields = line.strip().split('\t')
        
        if len(fields) < 8:
            return None
        
        chrom = fields[0]
        if not chrom.startswith('chr'):
            chrom = f'chr{chrom}'
        
        pos = int(fields[1])
        rs_id = fields[2] if fields[2] != '.' else None
        ref = fields[3]
        alts = fields[4].split(',')
        info = self._parse_info_field(fields[7])
        
        # Extract clinical significance
        clnsig = info.get('CLNSIG', ['uncertain_significance'])[0]
        clnsig_parts = clnsig.split('/')
        clinical_significance = clnsig_parts[0].lower().replace(' ', '_')
        
        # Extract gene symbol
        gene = info.get('GENEINFO', [None])[0]
        if gene and ':' in gene:
            gene = gene.split(':')[0]
        
        # Extract conditions
        conditions = []
        clndn = info.get('CLNDN', [])
        for condition_str in clndn:
            if condition_str and condition_str != '.':
                conditions.append(ClinicalCondition(name=condition_str))
        
        # Clinical annotations
        clnrevstat = info.get('CLNREVSTAT', ['no_assertion_provided'])[0]
        stars = self._review_status_to_stars(clnrevstat)
        
        clinical_annotations = ClinicalAnnotation(
            review_status=clnrevstat,
            stars=stars
        )
        
        # Functional impact
        mc = info.get('MC', [])
        consequence = mc[0].split('|')[0] if mc else 'unknown'
        
        functional_impact = FunctionalImpact(
            consequence=consequence
        )
        
        # Create SNP
        snp = ClinicalSNP(
            snp_id=rs_id or f"{chrom}:{pos}",
            chromosome=chrom,
            position=pos,
            ref_allele=ref,
            alt_alleles=alts,
            gene=gene,
            clinical_significance=clinical_significance,
            conditions=conditions,
            clinical_annotations=clinical_annotations,
            functional_impact=functional_impact,
            sources={
                'clinvar_id': info.get('CLNVI', [''])[0],
                'dbsnp_id': rs_id
            }
        )
        
        return snp
    
    def _parse_info_field(self, info_str: str) -> Dict[str, List[str]]:
        """Parse VCF INFO field"""
        info_dict = {}
        for item in info_str.split(';'):
            if '=' in item:
                key, value = item.split('=', 1)
                info_dict[key] = value.split(',')
            else:
                info_dict[item] = [True]
        return info_dict
    
    def _review_status_to_stars(self, review_status: str) -> int:
        """Convert ClinVar review status to star rating"""
        star_mapping = {
            'practice_guideline': 4,
            'reviewed_by_expert_panel': 3,
            'criteria_provided,_multiple_submitters,_no_conflicts': 2,
            'criteria_provided,_conflicting_interpretations': 1,
            'criteria_provided,_single_submitter': 1,
            'no_assertion_provided': 0,
            'no_assertion_criteria_provided': 0
        }
        return star_mapping.get(review_status.lower(), 0)


class ClinicalDataAcquisition:
    """
    Main pipeline to acquire and merge clinical variant data
    
    Usage:
        acquisition = ClinicalDataAcquisition(genome_build='GRCh38')
        acquisition.download_all_sources()
        acquisition.build_database(output_path='data/clinical_snps_v1.0.0.json.gz')
    """
    
    def __init__(self, genome_build: str = "GRCh38", output_dir: str = "data"):
        self.genome_build = genome_build
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        self.clinvar = ClinVarDownloader(genome_build, str(self.raw_dir))
    
    def download_all_sources(self):
        """Download data from all sources"""
        logger.info("=" * 80)
        logger.info("CLINICAL DATA ACQUISITION PIPELINE")
        logger.info("=" * 80)
        
        # ClinVar (primary source)
        logger.info("\n[1/1] Downloading ClinVar...")
        clinvar_path = self.clinvar.download_vcf()
        
        logger.info("\n" + "=" * 80)
        logger.info("Download phase complete!")
        logger.info("=" * 80)
    
    def build_database(self, 
                      output_path: str = "data/clinical_snps_v1.0.0.json.gz",
                      filter_pathogenic: bool = True,
                      min_review_stars: int = 1) -> Path:
        """
        Build unified clinical SNP database
        
        Args:
            output_path: Where to save the database
            filter_pathogenic: Only include pathogenic variants
            min_review_stars: Minimum ClinVar review stars
            
        Returns:
            Path to created database
        """
        logger.info("=" * 80)
        logger.info("BUILDING CLINICAL SNP DATABASE")
        logger.info("=" * 80)
        
        builder = ClinicalDatabaseBuilder(genome_build=self.genome_build)
        
        # Add ClinVar variants
        logger.info("\n[1/1] Processing ClinVar variants...")
        clinvar_vcf = self.raw_dir / f"clinvar_{self.genome_build}.vcf.gz"
        
        if clinvar_vcf.exists():
            clinvar_snps = self.clinvar.parse_vcf(
                clinvar_vcf,
                filter_pathogenic=filter_pathogenic,
                min_review_stars=min_review_stars
            )
            
            logger.info(f"Adding {len(clinvar_snps)} ClinVar SNPs to database...")
            for snp in clinvar_snps:
                builder.add_snp(snp)
        else:
            logger.warning("ClinVar VCF not found. Run download_all_sources() first.")
        
        # Save database
        logger.info(f"\nSaving database to {output_path}...")
        output_path = Path(output_path)
        builder.save(output_path, compress=True)
        
        logger.info("\n" + "=" * 80)
        logger.info("DATABASE BUILD COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Database saved to: {output_path}")
        if output_path.exists():
            logger.info(f"Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        
        return output_path


# CLI Interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and build clinical SNP database")
    parser.add_argument("--genome-build", default="GRCh38", choices=["GRCh38", "GRCh37"],
                       help="Genome build (default: GRCh38)")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--download-only", action="store_true", help="Only download, don't build")
    parser.add_argument("--build-only", action="store_true", help="Only build from existing data")
    parser.add_argument("--pathogenic-only", action="store_true", help="Only include pathogenic variants")
    parser.add_argument("--min-stars", type=int, default=1, help="Minimum ClinVar review stars")
    
    args = parser.parse_args()
    
    acquisition = ClinicalDataAcquisition(
        genome_build=args.genome_build,
        output_dir=args.output_dir
    )
    
    if not args.build_only:
        acquisition.download_all_sources()
    
    if not args.download_only:
        db_path = acquisition.build_database(
            output_path=f"{args.output_dir}/clinical_snps_v1.0.0.json.gz",
            filter_pathogenic=args.pathogenic_only,
            min_review_stars=args.min_stars
        )
        
        # Print summary
        from genomevault.clinical_db.database import ClinicalSNPDatabase
        db = ClinicalSNPDatabase(str(db_path))
        stats = db.get_statistics()
        
        print("\n" + "=" * 80)
        print("DATABASE STATISTICS")
        print("=" * 80)
        for key, value in stats.items():
            print(f"{key:25s}: {value}")
